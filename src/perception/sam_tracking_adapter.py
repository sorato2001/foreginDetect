"""SAM2-style railway intrusion tracker adapter for STEAD.

This module condenses the SAMTracking research pipeline into an optional STEAD
tracker adapter:

video frames -> YOLO object detection -> railway mask segmentation ->
SAM2/object-mask tracking -> optical-flow temporal smoothing ->
mask-IoU intrusion judgment -> sliding-window confirmation.

The adapter is intentionally dependency-tolerant. Heavy runtime dependencies
such as ultralytics, torch, SAM2 checkpoints, and model files are loaded lazily.
When they are missing, the adapter falls back to bbox masks and empty railway
masks so the rest of the STEAD demo still runs and records a degraded status.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.perception.detector import Detection
from src.perception.simple_iou_tracker import SimpleIOUTracker

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SAMTrackingConfig:
    """Configuration for the optional SAMTracking adapter."""

    object_model_path: str = "../RailwayIntrusion_Tracking_SAM2/weights/yolo11l.pt"
    track_model_path: str = "../RailwayIntrusion_Tracking_SAM2/weights/best.pt"
    sam2_config: str | None = None
    sam2_checkpoint: str | None = None
    device: str = "cuda"
    conf_threshold: float = 0.35
    target_labels: list[str] = field(default_factory=lambda: ["person", "cow", "sheep"])
    iou_threshold: float = 0.10
    window_size: int = 5
    confirm_count: int = 3
    use_optical_flow: bool = True
    fallback_to_bbox_mask: bool = True
    sample_every: int = 15
    track_mask_interval: int = 30
    imgsz: int = 640
    progress_interval: int = 10


@dataclass(slots=True)
class SAMFrameResult:
    """Per-frame SAMTracking judgment."""

    frame_index: int
    timestamp: float
    detections: list[dict[str, Any]]
    object_mask_count: int
    track_mask_available: bool
    max_iou: float
    suspicious: bool
    window_count: int
    alarm: bool


@dataclass(slots=True)
class SAMIntrusionEvent:
    """Confirmed intrusion interval from sliding-window logic."""

    event_id: int
    start_frame: int
    end_frame: int
    peak_iou: float
    frames_suspicious: list[int] = field(default_factory=list)


@dataclass(slots=True)
class SAMTrackingResult:
    """Structured output from SAMTrackingAdapter."""

    tracks: dict[int, list[Detection]]
    fps: float
    duration: float
    frames: list[SAMFrameResult] = field(default_factory=list)
    intrusion_events: list[SAMIntrusionEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "fps": self.fps,
            "duration": self.duration,
            "tracks": {
                str(track_id): [
                    {
                        "label": det.label,
                        "confidence": det.confidence,
                        "bbox": det.bbox,
                        "frame_index": det.frame_index,
                        "timestamp": det.timestamp,
                    }
                    for det in history
                ]
                for track_id, history in self.tracks.items()
            },
            "frames": [asdict(frame) for frame in self.frames],
            "intrusion_events": [asdict(event) for event in self.intrusion_events],
            "metadata": self.metadata,
        }


class SlidingWindowIntrusionJudge:
    """Mask-IoU intrusion judge with K-frame/N-hit confirmation."""

    def __init__(self, iou_threshold: float = 0.10, window_size: int = 5, confirm_count: int = 3) -> None:
        self.iou_threshold = iou_threshold
        self.window_size = max(1, window_size)
        self.confirm_count = max(1, confirm_count)
        self.history: deque[int] = deque(maxlen=self.window_size)
        self.events: list[SAMIntrusionEvent] = []
        self._active_event: SAMIntrusionEvent | None = None
        self._event_counter = 0

    @staticmethod
    def compute_iou(mask_a: Any, mask_b: Any) -> float:
        """Compute binary mask intersection-over-union."""
        try:
            import numpy as np
        except Exception:
            return 0.0
        a = np.asarray(mask_a).astype(bool)
        b = np.asarray(mask_b).astype(bool)
        if a.shape != b.shape:
            return 0.0
        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(intersection) / float(union) if union else 0.0

    def update(self, frame_index: int, max_iou: float) -> dict[str, Any]:
        """Update suspicious-frame history and event lifecycle."""
        suspicious = max_iou > self.iou_threshold
        self.history.append(1 if suspicious else 0)
        window_count = int(sum(self.history))
        alarm = window_count >= self.confirm_count

        if alarm and self._active_event is None:
            self._event_counter += 1
            self._active_event = SAMIntrusionEvent(
                event_id=self._event_counter,
                start_frame=frame_index,
                end_frame=frame_index,
                peak_iou=max_iou,
                frames_suspicious=[frame_index] if suspicious else [],
            )
        elif alarm and self._active_event is not None:
            self._active_event.end_frame = frame_index
            self._active_event.peak_iou = max(self._active_event.peak_iou, max_iou)
            if suspicious:
                self._active_event.frames_suspicious.append(frame_index)
        elif not alarm and self._active_event is not None:
            self.events.append(self._active_event)
            self._active_event = None

        return {"suspicious": suspicious, "window_count": window_count, "alarm": alarm}

    def get_events(self) -> list[SAMIntrusionEvent]:
        """Return completed and active events."""
        events = list(self.events)
        if self._active_event is not None:
            events.append(self._active_event)
        return events


class SAMTrackingAdapter:
    """Optional tracker adapter inspired by RailwayIntrusion_Tracking_SAM2."""

    def __init__(self, config: SAMTrackingConfig | None = None) -> None:
        self.config = config or SAMTrackingConfig()
        self._object_model: Any | None = None
        self._track_model: Any | None = None
        self._object_model_error: str | None = None
        self._track_model_error: str | None = None
        self._sam2_error: str | None = None
        self._runtime_device = self._resolve_device(self.config.device)
        self._smoother = _OpticalFlowMaskSmoother(enabled=self.config.use_optical_flow)
        self._load_models()

    def analyze_video(self, video_path: str, max_frames: int | None = 900) -> SAMTrackingResult:
        """Analyze a video and return STEAD-compatible tracks plus SAM diagnostics."""
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            logger.warning("SAMTracking unavailable because OpenCV/numpy import failed: %s", exc)
            return SAMTrackingResult(
                tracks={},
                fps=0.0,
                duration=0.0,
                metadata=self._metadata(degraded=True, error=f"{type(exc).__name__}: {exc}"),
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("SAMTracking cannot open video=%s", video_path)
            return SAMTrackingResult(
                tracks={},
                fps=0.0,
                duration=0.0,
                metadata=self._metadata(degraded=True, error="cannot_open_video"),
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_every = max(1, int(self.config.sample_every))
        track_mask_interval = max(1, int(self.config.track_mask_interval))
        tracker = SimpleIOUTracker(iou_threshold=0.3)
        judge = SlidingWindowIntrusionJudge(
            iou_threshold=self.config.iou_threshold,
            window_size=self.config.window_size,
            confirm_count=self.config.confirm_count,
        )
        frames: list[SAMFrameResult] = []
        frame_index = 0
        processed = 0
        track_mask_seen = False
        detection_seen = False
        cached_track_mask: Any | None = None
        logger.info(
            "SAMTracking analyze start: video=%s fps=%.3f total_frames=%s max_frames=%s sample_every=%s track_mask_interval=%s device=%s imgsz=%s",
            video_path,
            fps,
            total_frames,
            max_frames,
            sample_every,
            track_mask_interval,
            self._runtime_device,
            self.config.imgsz,
        )

        while True:
            if max_frames and frame_index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % sample_every != 0:
                frame_index += 1
                continue

            timestamp = frame_index / fps if fps > 0 else 0.0
            detections = self.detect_objects(frame, frame_index, timestamp)
            detection_seen = detection_seen or bool(detections)
            tracks = tracker.update(detections)
            if cached_track_mask is None or frame_index % track_mask_interval == 0:
                cached_track_mask = self.detect_track_mask(frame)
            track_mask = cached_track_mask
            track_mask_seen = track_mask_seen or bool(track_mask is not None and np.asarray(track_mask).sum() > 0)
            object_masks = self.segment_objects(frame, detections)
            if self.config.use_optical_flow and object_masks:
                object_masks = [self._smoother.smooth(frame, mask) for mask in object_masks]
            max_iou = max((judge.compute_iou(mask, track_mask) for mask in object_masks), default=0.0) if track_mask is not None else 0.0
            state = judge.update(frame_index, max_iou)
            frames.append(
                SAMFrameResult(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    detections=[_detection_to_dict(det) for det in detections],
                    object_mask_count=len(object_masks),
                    track_mask_available=track_mask is not None and bool(np.asarray(track_mask).sum() > 0),
                    max_iou=max_iou,
                    suspicious=bool(state["suspicious"]),
                    window_count=int(state["window_count"]),
                    alarm=bool(state["alarm"]),
                )
            )
            processed += 1
            if self.config.progress_interval > 0 and processed % self.config.progress_interval == 0:
                logger.info(
                    "SAMTracking progress: processed=%s frame=%s/%s tracks=%s detections_seen=%s track_mask_seen=%s last_iou=%.4f",
                    processed,
                    frame_index,
                    total_frames or "?",
                    len(tracks),
                    detection_seen,
                    track_mask_seen,
                    max_iou,
                )
            frame_index += 1

        cap.release()
        duration = frame_index / fps if fps > 0 else 0.0
        degraded = self._object_model is None or self._track_model is None or self._sam2_error is not None
        metadata = self._metadata(degraded=degraded)
        metadata.update(
            {
                "processed_frames": processed,
                "total_frames": total_frames,
                "sample_every": sample_every,
                "track_mask_interval": track_mask_interval,
                "track_mask_seen": track_mask_seen,
                "detections_seen": detection_seen,
                "fallback_to_bbox_mask": self.config.fallback_to_bbox_mask,
                "sam2_mode": "placeholder_or_bbox_mask",
            }
        )
        return SAMTrackingResult(
            tracks=tracks,
            fps=fps,
            duration=duration,
            frames=frames,
            intrusion_events=judge.get_events(),
            metadata=metadata,
        )

    def detect_objects(self, frame: Any, frame_index: int, timestamp: float) -> list[Detection]:
        """Detect person/cow/sheep with YOLO when available."""
        if self._object_model is None:
            return []
        try:
            results = self._object_model.predict(
                source=frame,
                conf=self.config.conf_threshold,
                imgsz=self.config.imgsz,
                device=self._runtime_device,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("SAMTracking object detection failed: %s", exc)
            return []
        names = getattr(self._object_model, "names", {}) or {}
        detections: list[Detection] = []
        if not results or getattr(results[0], "boxes", None) is None:
            return detections
        targets = set(self.config.target_labels)
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            label = str(names.get(cls_id, cls_id))
            if targets and label not in targets:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(Detection(label, float(box.conf.item()), [x1, y1, x2, y2], frame_index, timestamp))
        return detections

    def detect_track_mask(self, frame: Any) -> Any | None:
        """Segment railway/track region with best.pt when available."""
        if self._track_model is None:
            return None
        try:
            import cv2
            import numpy as np

            results = self._track_model.predict(
                source=frame,
                conf=self.config.conf_threshold,
                imgsz=self.config.imgsz,
                device=self._runtime_device,
                verbose=False,
            )
            if not results or getattr(results[0], "masks", None) is None or results[0].masks is None:
                return None
            masks = results[0].masks.data
            if len(masks) == 0:
                return None
            merged = np.zeros(frame.shape[:2], dtype=np.uint8)
            for mask_tensor in masks:
                mask = mask_tensor.detach().cpu().numpy().astype("float32")
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                merged = np.maximum(merged, (mask >= 0.5).astype(np.uint8))
            return merged
        except Exception as exc:
            logger.warning("SAMTracking track segmentation failed: %s", exc)
            return None

    def segment_objects(self, frame: Any, detections: list[Detection]) -> list[Any]:
        """Return object masks; falls back to bbox masks until SAM2 is configured."""
        if not detections:
            return []
        if self.config.fallback_to_bbox_mask:
            return [bbox_to_mask(frame.shape[:2], det.bbox) for det in detections]
        return []

    def _load_models(self) -> None:
        """Lazy-load YOLO models and record missing SAM2 as degraded metadata."""
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self._object_model_error = f"ultralytics_unavailable: {exc}"
            self._track_model_error = self._object_model_error
            logger.warning("SAMTracking YOLO unavailable, adapter will degrade: %s", exc)
            self._sam2_error = "sam2_not_loaded"
            return

        self._object_model = self._try_load_yolo(YOLO, self.config.object_model_path, "object")
        self._track_model = self._try_load_yolo(YOLO, self.config.track_model_path, "track")
        self._sam2_error = self._probe_sam2()

    def _try_load_yolo(self, yolo_cls: Any, model_path: str, role: str) -> Any | None:
        path = Path(model_path)
        try:
            model = yolo_cls(model_path)
            logger.info("Loaded SAMTracking %s YOLO model: %s", role, model_path)
            return model
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if role == "object":
                self._object_model_error = error
            else:
                self._track_model_error = error
            missing_note = "missing file; " if not path.exists() and not _looks_like_builtin_model(model_path) else ""
            logger.warning("SAMTracking %s YOLO model unavailable (%s%s)", role, missing_note, error)
            return None

    def _probe_sam2(self) -> str | None:
        """Check SAM2 availability without making it mandatory."""
        if not self.config.sam2_config and not self.config.sam2_checkpoint:
            return "sam2_not_configured"
        try:
            import sam2  # noqa: F401
        except Exception as exc:
            return f"sam2_unavailable: {exc}"
        return "sam2_streaming_memory_not_initialized_in_adapter"

    def _metadata(self, degraded: bool, error: str | None = None) -> dict[str, Any]:
        return {
            "adapter": "sam_tracking",
            "pipeline": [
                "video_frames",
                "yolo11_object_detection",
                "best_pt_rail_segmentation",
                "sam2_or_bbox_mask_tracking",
                "optical_flow_temporal_smoothing",
                "mask_iou_intrusion_judgment",
                "sliding_window_confirmation",
            ],
            "degraded": degraded,
            "error": error,
            "object_model_path": self.config.object_model_path,
            "track_model_path": self.config.track_model_path,
            "object_model_loaded": self._object_model is not None,
            "track_model_loaded": self._track_model is not None,
            "object_model_error": self._object_model_error,
            "track_model_error": self._track_model_error,
            "sam2_status": self._sam2_error or "available",
            "requested_device": self.config.device,
            "runtime_device": self._runtime_device,
            "target_labels": self.config.target_labels,
            "iou_threshold": self.config.iou_threshold,
            "window_size": self.config.window_size,
            "confirm_count": self.config.confirm_count,
            "use_optical_flow": self.config.use_optical_flow,
            "imgsz": self.config.imgsz,
        }

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Use CPU automatically when CUDA is requested but unavailable."""
        if requested.lower().startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning("SAMTracking requested CUDA but CUDA is unavailable; using CPU")
                    return "cpu"
            except Exception:
                logger.warning("SAMTracking cannot check CUDA availability; using CPU")
                return "cpu"
        return requested


class _OpticalFlowMaskSmoother:
    """Small Farneback-based binary mask smoother with graceful fallback."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._prev_frame: Any | None = None
        self._prev_mask: Any | None = None

    def smooth(self, frame: Any, mask: Any) -> Any:
        """Warp and blend the previous mask into the current mask."""
        if not self.enabled:
            return mask
        try:
            import cv2
            import numpy as np
        except Exception:
            return mask
        if self._prev_frame is None or self._prev_mask is None:
            self._prev_frame = frame.copy()
            self._prev_mask = mask.copy()
            return mask
        try:
            prev_gray = cv2.cvtColor(self._prev_frame, cv2.COLOR_BGR2GRAY) if self._prev_frame.ndim == 3 else self._prev_frame
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = flow.shape[:2]
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            warped = cv2.remap(
                self._prev_mask.astype(np.float32),
                x_coords + flow[..., 0],
                y_coords + flow[..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            fused = (((mask.astype(np.float32) * 0.7) + (warped * 0.3)) >= 0.5).astype(np.uint8)
            self._prev_frame = frame.copy()
            self._prev_mask = fused.copy()
            return fused
        except Exception:
            self._prev_frame = frame.copy()
            self._prev_mask = mask.copy()
            return mask


def bbox_to_mask(frame_shape: tuple[int, int] | list[int], bbox: list[float]) -> Any:
    """Convert a detection bbox to a binary mask."""
    import numpy as np

    height, width = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    mask = np.zeros((height, width), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def _detection_to_dict(det: Detection) -> dict[str, Any]:
    return {
        "label": det.label,
        "confidence": det.confidence,
        "bbox": det.bbox,
        "frame_index": det.frame_index,
        "timestamp": det.timestamp,
    }


def _looks_like_builtin_model(model_path: str) -> bool:
    return "/" not in model_path and "\\" not in model_path and model_path.endswith(".pt")
