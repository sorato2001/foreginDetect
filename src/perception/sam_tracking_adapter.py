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
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.perception.detector import Detection
from src.perception.simple_iou_tracker import SimpleIOUTracker

logger = logging.getLogger(__name__)


def _box_iou(box_a: Any, box_b: Any) -> float:
    """Compute IoU for two xyxy boxes."""
    try:
        import numpy as np
    except Exception:
        return 0.0
    a = np.asarray(box_a, dtype=float)
    b = np.asarray(box_b, dtype=float)
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def det_to_track_id(det: Detection, tracks: dict[int, list[Detection]]) -> int | None:
    """Find the current track id for a detection."""
    for track_id, history in tracks.items():
        if not history:
            continue
        last = history[-1]
        if last.frame_index == det.frame_index and last.label == det.label and last.bbox == det.bbox:
            return track_id
    best_id: int | None = None
    best_score = 0.0
    for track_id, history in tracks.items():
        if not history or history[-1].label != det.label:
            continue
        score = _box_iou(det.bbox, history[-1].bbox)
        if score > best_score:
            best_score = score
            best_id = track_id
    return best_id


@dataclass(slots=True)
class SAMTrackingConfig:
    """Configuration for the optional SAMTracking adapter."""

    object_model_path: str = "../RailwayIntrusion_Tracking_SAM2/weights/yolo11l.pt"
    track_model_path: str = "../RailwayIntrusion_Tracking_SAM2/weights/best.pt"
    rule_region_source: str = "sam_track"
    rule_region_source_requested: str = "sam_track"
    sam2_config: str | None = None
    sam2_checkpoint: str | None = None
    device: str = "cuda"
    conf_threshold: float = 0.35
    target_labels: list[str] = field(default_factory=lambda: ["person", "cow", "sheep"])
    track_mask_labels: list[str] = field(default_factory=list)
    iou_threshold: float = 0.10
    object_overlap_threshold: float = 0.15
    window_size: int = 5
    confirm_count: int = 3
    use_optical_flow: bool = True
    fallback_to_bbox_mask: bool = True
    sam2_enabled: bool = True
    sam2_scan_frames: int = 30
    sam2_prompt_mode: str = "bounding_box"
    sam2_new_object_iou_threshold: float = 0.3
    temporal_mode: str = "fast"
    sample_every: int = 15
    track_mask_interval: int = 30
    imgsz: int = 640
    progress_interval: int = 10
    output_dir: str | None = None
    guard_net_text_prompt: str = (
        "entire continuous black metal chain link fence"
    )
    guard_net_box_threshold: float = 0.15
    guard_net_text_threshold: float = 0.15
    guard_net_max_box_area_ratio: float = 0.6
    guard_net_candidate_count: int = 12
    guard_net_crop_roi: str | None = None
    guard_net_selection_mode: str = "best"
    guard_net_target_area_ratio: float = 0.55
    guard_net_mask_output_mode: str = "sam"
    guard_net_continuous_band_margin: int = 4
    guard_net_continuous_band_endpoint_source: str = "largest-component"
    guard_net_save_selected_sam_mask: bool = False


@dataclass(slots=True)
class SAMFrameResult:
    """Per-frame SAMTracking judgment."""

    frame_index: int
    timestamp: float
    detections: list[dict[str, Any]]
    object_mask_count: int
    track_mask_available: bool
    max_iou: float
    max_object_overlap: float
    suspicious: bool
    window_count: int
    alarm: bool
    rule_region_source: str = "sam_track"
    rule_region_available: bool = False
    rule_region_mask_area: int = 0
    track_mask_contours: list[list[list[int]]] = field(default_factory=list)
    object_mask_contours: list[list[list[list[int]]]] = field(default_factory=list)


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

    def __init__(
        self,
        iou_threshold: float = 0.10,
        object_overlap_threshold: float = 0.15,
        window_size: int = 5,
        confirm_count: int = 3,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.object_overlap_threshold = object_overlap_threshold
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

    @staticmethod
    def compute_object_overlap(object_mask: Any, track_mask: Any) -> float:
        """Compute how much of an object mask lies inside the track mask."""
        try:
            import numpy as np
        except Exception:
            return 0.0
        obj = np.asarray(object_mask).astype(bool)
        track = np.asarray(track_mask).astype(bool)
        if obj.shape != track.shape:
            return 0.0
        obj_area = obj.sum()
        if obj_area == 0:
            return 0.0
        intersection = np.logical_and(obj, track).sum()
        return float(intersection) / float(obj_area)

    def update(self, frame_index: int, max_iou: float, max_object_overlap: float = 0.0) -> dict[str, Any]:
        """Update suspicious-frame history and event lifecycle."""
        suspicious = max_iou > self.iou_threshold or max_object_overlap >= self.object_overlap_threshold
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
        self._sam2_tracker: _SAM2VideoMemoryTracker | None = None
        self._sam2_temp_video: Path | None = None
        self._smoothers: dict[int, _OpticalFlowProbabilitySmoother] = {}
        self._guard_net_result: Any | None = None
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
        temporal_mode = (self.config.temporal_mode or "fast").strip().lower()
        if temporal_mode not in {"fast", "faithful", "reference"}:
            logger.warning("Unknown SAMTracking temporal_mode=%s, using fast", self.config.temporal_mode)
            temporal_mode = "fast"
        if temporal_mode == "reference":
            temporal_mode = "faithful"
        sample_every = 1 if temporal_mode == "faithful" else max(1, int(self.config.sample_every))
        track_mask_interval = 1 if temporal_mode == "faithful" else int(self.config.track_mask_interval)
        track_mask_enabled = track_mask_interval > 0
        if track_mask_enabled:
            track_mask_interval = max(1, track_mask_interval)
        tracker = SimpleIOUTracker(iou_threshold=0.3)
        judge = SlidingWindowIntrusionJudge(
            iou_threshold=self.config.iou_threshold,
            object_overlap_threshold=self.config.object_overlap_threshold,
            window_size=self.config.window_size,
            confirm_count=self.config.confirm_count,
        )
        frames: list[SAMFrameResult] = []
        frame_index = 0
        processed = 0
        track_mask_seen = False
        detection_seen = False
        cached_track_mask: Any | None = None
        if self.config.sam2_enabled:
            self._initialize_sam2_video(video_path, fps, max_frames, total_frames=total_frames)
        logger.info(
            "SAMTracking analyze start: video=%s fps=%.3f total_frames=%s max_frames=%s temporal_mode=%s sample_every=%s track_mask_interval=%s track_mask_enabled=%s device=%s imgsz=%s",
            video_path,
            fps,
            total_frames,
            max_frames,
            temporal_mode,
            sample_every,
            track_mask_interval,
            track_mask_enabled,
            self._runtime_device,
            self.config.imgsz,
        )

        while True:
            if max_frames and frame_index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_index / fps if fps > 0 else 0.0
            should_detect_objects = temporal_mode == "faithful" or frame_index % sample_every == 0
            detections = self.detect_objects(frame, frame_index, timestamp) if should_detect_objects else []
            detection_seen = detection_seen or bool(detections)
            if detections:
                tracks = tracker.update(detections)
            else:
                tracks = tracker.tracks
            if track_mask_enabled and (cached_track_mask is None or frame_index % track_mask_interval == 0):
                cached_track_mask = self.detect_track_mask(frame)
            track_mask = cached_track_mask
            track_mask_seen = track_mask_seen or bool(track_mask is not None and np.asarray(track_mask).sum() > 0)
            object_ids = [det_to_track_id(det, tracks) for det in detections]
            object_masks, mask_object_ids = self._segment_objects_with_ids(
                frame,
                detections,
                frame_index=frame_index,
                object_ids=object_ids,
            )
            if self.config.use_optical_flow and object_masks:
                object_masks = [
                    self._smooth_object_mask(frame, mask, mask_object_ids[index] if index < len(mask_object_ids) else index)
                    for index, mask in enumerate(object_masks)
                ]
            if track_mask is not None:
                max_iou = max((judge.compute_iou(mask, track_mask) for mask in object_masks), default=0.0)
                max_object_overlap = max((judge.compute_object_overlap(mask, track_mask) for mask in object_masks), default=0.0)
            else:
                max_iou = 0.0
                max_object_overlap = 0.0
            state = judge.update(frame_index, max_iou, max_object_overlap=max_object_overlap)
            track_mask_contours = mask_to_contours(track_mask) if track_mask is not None else []
            object_mask_contours = [mask_to_contours(mask) for mask in object_masks]
            rule_region_available = track_mask is not None and bool(np.asarray(track_mask).sum() > 0)
            rule_region_mask_area = int(np.asarray(track_mask).sum()) if track_mask is not None else 0
            frames.append(
                SAMFrameResult(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    detections=[_detection_to_dict(det) for det in detections],
                    object_mask_count=len(object_masks),
                    track_mask_available=rule_region_available,
                    max_iou=max_iou,
                    max_object_overlap=max_object_overlap,
                    suspicious=bool(state["suspicious"]),
                    window_count=int(state["window_count"]),
                    alarm=bool(state["alarm"]),
                    rule_region_source=self.config.rule_region_source,
                    rule_region_available=rule_region_available,
                    rule_region_mask_area=rule_region_mask_area,
                    track_mask_contours=track_mask_contours,
                    object_mask_contours=object_mask_contours,
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
        needs_track_model = (self.config.rule_region_source or "sam_track") == "sam_track"
        degraded = self._object_model is None or (needs_track_model and self._track_model is None) or self._sam2_error is not None
        metadata = self._metadata(degraded=degraded)
        metadata.update(
            {
                "processed_frames": processed,
                "total_frames": total_frames,
                "temporal_mode": temporal_mode,
                "sample_every": sample_every,
                "track_mask_interval": track_mask_interval,
                "track_mask_enabled": track_mask_enabled,
                "track_mask_seen": track_mask_seen,
                "rule_region_available": track_mask_seen,
                "detections_seen": detection_seen,
                "fallback_to_bbox_mask": self.config.fallback_to_bbox_mask,
                "sam2_mode": "streaming_memory" if self._sam2_tracker is not None else "bbox_mask",
            }
        )
        self._cleanup_sam2_temp_video()
        return SAMTrackingResult(
            tracks=tracks,
            fps=fps,
            duration=duration,
            frames=frames,
            intrusion_events=judge.get_events(),
            metadata=metadata,
        )

    def analyze_image(self, image_path: str) -> SAMTrackingResult:
        """Analyze one image with the same object/track-mask intrusion logic."""
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            logger.warning("SAMTracking unavailable because OpenCV/numpy import failed: %s", exc)
            return SAMTrackingResult(
                tracks={},
                fps=0.0,
                duration=1.0,
                metadata=self._metadata(degraded=True, error=f"{type(exc).__name__}: {exc}"),
            )

        frame = cv2.imread(image_path)
        if frame is None:
            logger.warning("SAMTracking cannot read image=%s", image_path)
            return SAMTrackingResult(
                tracks={},
                fps=0.0,
                duration=1.0,
                metadata=self._metadata(degraded=True, error="cannot_read_image"),
            )

        frame_index = 0
        timestamp = 0.0
        detections = self.detect_objects(frame, frame_index, timestamp)
        tracks = {track_id + 1: [det] for track_id, det in enumerate(detections)}
        track_mask = self.detect_track_mask(frame)
        object_ids = list(tracks.keys())
        object_masks = self.segment_objects(frame, detections, frame_index=frame_index, object_ids=object_ids)
        judge = SlidingWindowIntrusionJudge(
            iou_threshold=self.config.iou_threshold,
            object_overlap_threshold=self.config.object_overlap_threshold,
            window_size=1,
            confirm_count=1,
        )
        track_mask_seen = bool(track_mask is not None and np.asarray(track_mask).sum() > 0)
        detection_seen = bool(detections)
        if track_mask is not None:
            max_iou = max((judge.compute_iou(mask, track_mask) for mask in object_masks), default=0.0)
            max_object_overlap = max((judge.compute_object_overlap(mask, track_mask) for mask in object_masks), default=0.0)
        else:
            max_iou = 0.0
            max_object_overlap = 0.0
        state = judge.update(frame_index, max_iou, max_object_overlap=max_object_overlap)
        rule_region_mask_area = int(np.asarray(track_mask).sum()) if track_mask is not None else 0
        frames = [
            SAMFrameResult(
                frame_index=frame_index,
                timestamp=timestamp,
                detections=[_detection_to_dict(det) for det in detections],
                object_mask_count=len(object_masks),
                track_mask_available=track_mask_seen,
                max_iou=max_iou,
                max_object_overlap=max_object_overlap,
                suspicious=bool(state["suspicious"]),
                window_count=int(state["window_count"]),
                alarm=bool(state["alarm"]),
                rule_region_source=self.config.rule_region_source,
                rule_region_available=track_mask_seen,
                rule_region_mask_area=rule_region_mask_area,
                track_mask_contours=mask_to_contours(track_mask) if track_mask is not None else [],
                object_mask_contours=[mask_to_contours(mask) for mask in object_masks],
            )
        ]
        needs_track_model = (self.config.rule_region_source or "sam_track") == "sam_track"
        degraded = self._object_model is None or (needs_track_model and self._track_model is None)
        metadata = self._metadata(degraded=degraded)
        metadata.update(
            {
                "input_type": "image",
                "processed_frames": 1,
                "total_frames": 1,
                "sample_every": 1,
                "track_mask_interval": 1,
                "track_mask_enabled": True,
                "track_mask_seen": track_mask_seen,
                "rule_region_available": track_mask_seen,
                "detections_seen": detection_seen,
                "fallback_to_bbox_mask": self.config.fallback_to_bbox_mask,
                "sam2_mode": "single_image_bbox_or_static_mask",
            }
        )
        return SAMTrackingResult(
            tracks=tracks,
            fps=0.0,
            duration=1.0,
            frames=frames,
            intrusion_events=judge.get_events(),
            metadata=metadata,
        )

    def detect_objects(self, frame: Any, frame_index: int, timestamp: float) -> list[Detection]:
        """Detect person/cow/sheep with YOLO when available."""
        if self._object_model is None:
            return []
        try:
            start = time.perf_counter()
            logger.info("SAMTracking object predict start: frame=%s device=%s imgsz=%s", frame_index, self._runtime_device, self.config.imgsz)
            results = self._object_model.predict(
                source=frame,
                conf=self.config.conf_threshold,
                imgsz=self.config.imgsz,
                device=self._runtime_device,
                verbose=False,
            )
            logger.info("SAMTracking object predict done: frame=%s elapsed=%.3fs", frame_index, time.perf_counter() - start)
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
        source = (self.config.rule_region_source or "sam_track").strip().lower()
        if source == "yaml":
            return None
        if source == "guard_net":
            return self._detect_guard_net_mask(frame)
        if self._track_model is None:
            return None
        try:
            import cv2
            import numpy as np

            start = time.perf_counter()
            logger.info("SAMTracking track-mask predict start: device=%s imgsz=%s", self._runtime_device, self.config.imgsz)
            results = self._track_model.predict(
                source=frame,
                conf=self.config.conf_threshold,
                imgsz=self.config.imgsz,
                device=self._runtime_device,
                verbose=False,
            )
            logger.info("SAMTracking track-mask predict done: elapsed=%.3fs", time.perf_counter() - start)
            merged = np.zeros(frame.shape[:2], dtype=np.uint8)
            has_any = False
            allowed_labels = {str(item) for item in self.config.track_mask_labels}
            for result in results or []:
                names = getattr(result, "names", None) or getattr(self._track_model, "names", {}) or {}
                if getattr(result, "masks", None) is not None and result.masks is not None:
                    masks = result.masks.data
                    classes = []
                    if getattr(result, "boxes", None) is not None and result.boxes is not None and getattr(result.boxes, "cls", None) is not None:
                        classes = [int(item) for item in result.boxes.cls.detach().cpu().numpy().tolist()]
                    for mask_index, mask_tensor in enumerate(masks):
                        cls_id = classes[mask_index] if mask_index < len(classes) else None
                        label = str(names.get(cls_id, cls_id)) if cls_id is not None else ""
                        if not _class_allowed(label, cls_id, allowed_labels):
                            continue
                        mask = mask_tensor.detach().cpu().numpy().astype("float32")
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                        binary_mask = (mask >= 0.5).astype(np.uint8)
                        if binary_mask.sum() == 0:
                            continue
                        merged = np.maximum(merged, binary_mask)
                        has_any = True
                elif getattr(result, "boxes", None) is not None and result.boxes is not None:
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    confs = result.boxes.conf.detach().cpu().numpy()
                    classes = result.boxes.cls.detach().cpu().numpy() if getattr(result.boxes, "cls", None) is not None else [None] * len(boxes)
                    for bbox, conf, cls_id in zip(boxes, confs, classes):
                        if float(conf) < self.config.conf_threshold:
                            continue
                        class_id = int(cls_id) if cls_id is not None else None
                        label = str(names.get(class_id, class_id)) if class_id is not None else ""
                        if not _class_allowed(label, class_id, allowed_labels):
                            continue
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if x2 > x1 and y2 > y1:
                            merged[y1:y2, x1:x2] = 1
                            has_any = True
            return merged if has_any else None
        except Exception as exc:
            logger.warning("SAMTracking track segmentation failed: %s", exc)
            return None

    def _detect_guard_net_mask(self, frame: Any) -> Any | None:
        """Generate or reuse a GroundingDINO + SAM2 guard-net rule mask."""
        if self._guard_net_result is not None:
            return getattr(self._guard_net_result, "mask", None)
        try:
            from src.perception.guard_net_region_adapter import GuardNetRegionAdapter, GuardNetRegionConfig

            config = GuardNetRegionConfig(
                text_prompt=self.config.guard_net_text_prompt,
                box_threshold=self.config.guard_net_box_threshold,
                text_threshold=self.config.guard_net_text_threshold,
                max_box_area_ratio=self.config.guard_net_max_box_area_ratio,
                candidate_count=self.config.guard_net_candidate_count,
                crop_roi=self.config.guard_net_crop_roi,
                selection_mode=self.config.guard_net_selection_mode,
                target_area_ratio=self.config.guard_net_target_area_ratio,
                mask_output_mode=self.config.guard_net_mask_output_mode,
                continuous_band_margin=self.config.guard_net_continuous_band_margin,
                continuous_band_endpoint_source=self.config.guard_net_continuous_band_endpoint_source,
                save_selected_sam_mask=self.config.guard_net_save_selected_sam_mask,
            )
            output_dir = Path(self.config.output_dir) / "guard_net" if self.config.output_dir else None
            adapter = GuardNetRegionAdapter(
                config,
                sam2_config=self.config.sam2_config,
                sam2_checkpoint=self.config.sam2_checkpoint,
                device=self._runtime_device,
                output_dir=output_dir,
            )
            self._guard_net_result = adapter.analyze_frame(frame, frame_index=0)
            if not self._guard_net_result.available:
                logger.warning("Guard-net rule region unavailable: %s", self._guard_net_result.fallback_reason)
                return None
            logger.info(
                "Guard-net rule region ready: candidate=%s score=%s mode=%s",
                self._guard_net_result.selected_candidate_index,
                self._guard_net_result.selection_score,
                self._guard_net_result.mask_output_mode,
            )
            return self._guard_net_result.mask
        except Exception as exc:
            logger.warning("Guard-net rule region failed: %s", exc)
            self._guard_net_result = None
            return None

    def segment_objects(
        self,
        frame: Any,
        detections: list[Detection],
        frame_index: int = 0,
        object_ids: list[int | None] | None = None,
    ) -> list[Any]:
        """Return SAM2 object masks, falling back to bbox masks if needed."""
        masks, _mask_ids = self._segment_objects_with_ids(frame, detections, frame_index=frame_index, object_ids=object_ids)
        return masks

    def _segment_objects_with_ids(
        self,
        frame: Any,
        detections: list[Detection],
        frame_index: int = 0,
        object_ids: list[int | None] | None = None,
    ) -> tuple[list[Any], list[int | None]]:
        """Return object masks and stable mask ids for temporal smoothing."""
        if not detections:
            if self._sam2_tracker is not None:
                try:
                    masks, _scores, sam_object_ids = self._sam2_tracker.track_frame(frame_index, None)
                    return masks, [int(item) for item in sam_object_ids]
                except Exception as exc:
                    self._sam2_error = f"sam2_track_failed: {type(exc).__name__}: {exc}"
                    logger.warning("SAM2 tracking failed at frame=%s, fallback to empty masks: %s", frame_index, exc)
            return [], []
        if self._sam2_tracker is not None:
            try:
                masks, _scores, sam_object_ids = self._sam2_tracker.track_frame(
                    frame_index,
                    self._detections_to_prompts(detections),
                )
                if masks:
                    return masks, [int(item) for item in sam_object_ids]
                logger.debug("SAM2 returned no masks at frame=%s ids=%s", frame_index, sam_object_ids)
            except Exception as exc:
                self._sam2_error = f"sam2_track_failed: {type(exc).__name__}: {exc}"
                logger.warning("SAM2 tracking failed at frame=%s, fallback to bbox masks: %s", frame_index, exc)
        if self.config.fallback_to_bbox_mask:
            return [bbox_to_mask(frame.shape[:2], det.bbox) for det in detections], list(object_ids or [None] * len(detections))
        return [], []

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
        if (self.config.rule_region_source or "sam_track") == "sam_track":
            self._track_model = self._try_load_yolo(YOLO, self.config.track_model_path, "track")
        else:
            self._track_model = None
            self._track_model_error = f"not_used_for_rule_region_source:{self.config.rule_region_source}"
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
        if not self.config.sam2_enabled:
            return "sam2_disabled"
        if not self.config.sam2_config or not self.config.sam2_checkpoint:
            return "sam2_not_configured"
        try:
            import decord  # noqa: F401
            from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
        except Exception as exc:
            return f"sam2_unavailable: {exc}"
        return None

    def _initialize_sam2_video(
        self,
        video_path: str,
        fps: float,
        max_frames: int | None,
        total_frames: int = 0,
    ) -> None:
        """Initialize SAM2 streaming memory tracker from first-frame YOLO prompts."""
        if self._sam2_error is not None or not self.config.sam2_enabled:
            return
        if not self.config.sam2_config or not self.config.sam2_checkpoint:
            self._sam2_error = "sam2_not_configured"
            return
        try:
            scan_prompts = self._scan_initial_prompts(video_path, fps, max_frames)
            sam2_video_path = self._prepare_sam2_video_source(video_path, max_frames, total_frames)
            tracker = _SAM2VideoMemoryTracker(
                sam2_config=self.config.sam2_config,
                sam2_checkpoint=self.config.sam2_checkpoint,
                device=self._runtime_device,
                prompt_mode=self.config.sam2_prompt_mode,
                scan_frames=self.config.sam2_scan_frames,
                new_obj_iou_thresh=self.config.sam2_new_object_iou_threshold,
            )
            tracker.initialize_video(sam2_video_path, scan_prompts=scan_prompts)
            self._sam2_tracker = tracker
            self._sam2_error = None
            logger.info("SAM2 streaming memory tracker initialized: prompts=%s video_source=%s", len(scan_prompts), sam2_video_path)
        except Exception as exc:
            self._sam2_tracker = None
            self._sam2_error = f"sam2_init_failed: {type(exc).__name__}: {exc}"
            self._cleanup_sam2_temp_video()
            logger.warning("SAM2 initialization failed, fallback to bbox masks: %s", exc)

    def _prepare_sam2_video_source(self, video_path: str, max_frames: int | None, total_frames: int = 0) -> str:
        """Return a short clip for SAM2 when analysis is frame-limited.

        SAM2's video predictor may initialize state for the whole source video.
        When STEAD only analyzes the first N frames, feeding SAM2 a short clip
        keeps CPU memory proportional to the actual analysis window.
        """
        if not max_frames or max_frames <= 0:
            return video_path
        if total_frames and max_frames >= total_frames:
            return video_path
        try:
            import cv2
        except Exception:
            return video_path

        source = Path(video_path)
        temp_dir = Path(tempfile.mkdtemp(prefix="stead_sam2_clip_"))
        temp_path = temp_dir / f"{source.stem}_first_{max_frames}.mp4"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return video_path
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            return video_path

        writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (width, height))
        written = 0
        while written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        writer.release()
        cap.release()
        if written <= 0 or not temp_path.exists() or temp_path.stat().st_size <= 0:
            return video_path
        self._sam2_temp_video = temp_path
        logger.info("SAM2 using frame-limited temp video: path=%s frames=%s/%s", temp_path, written, total_frames or "?")
        return str(temp_path)

    def _cleanup_sam2_temp_video(self) -> None:
        """Remove the temporary SAM2 clip after video analysis finishes."""
        temp_path = self._sam2_temp_video
        self._sam2_temp_video = None
        if temp_path is None:
            return
        try:
            parent = temp_path.parent
            if temp_path.exists():
                temp_path.unlink()
            if parent.exists():
                parent.rmdir()
        except Exception as exc:
            logger.debug("Could not remove SAM2 temp video %s: %s", temp_path, exc)

    def _scan_initial_prompts(self, video_path: str, fps: float, max_frames: int | None) -> list[dict[str, Any]]:
        """Scan initial frames with YOLO11l and convert boxes to SAM2 prompts."""
        try:
            import cv2
        except Exception:
            return []
        scan_limit = self.config.sam2_scan_frames
        if max_frames:
            scan_limit = min(scan_limit, max_frames)
        cap = cv2.VideoCapture(video_path)
        prompts: list[dict[str, Any]] = []
        frame_index = 0
        while frame_index < scan_limit:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / fps if fps > 0 else 0.0
            detections = self.detect_objects(frame, frame_index, timestamp)
            prompts.append(self._detections_to_prompts(detections))
            frame_index += 1
        cap.release()
        return prompts

    @staticmethod
    def _detections_to_prompts(detections: list[Detection]) -> dict[str, Any]:
        """Convert STEAD detections to SAM2 box prompts."""
        try:
            import numpy as np
        except Exception:
            return {"boxes": [], "labels": [], "class_names": []}
        if not detections:
            return {"boxes": np.empty((0, 4)), "labels": np.empty((0,)), "class_names": []}
        return {
            "boxes": np.array([det.bbox for det in detections], dtype=np.float32),
            "labels": np.arange(len(detections), dtype=np.int32),
            "class_names": [det.label for det in detections],
        }

    def _smooth_object_mask(self, frame: Any, mask: Any, object_id: int | None) -> Any:
        key = int(object_id if object_id is not None else -1)
        smoother = self._smoothers.setdefault(key, _OpticalFlowProbabilitySmoother(enabled=True))
        return smoother.smooth_mask(frame, mask)

    def _metadata(self, degraded: bool, error: str | None = None) -> dict[str, Any]:
        requested = self.config.rule_region_source_requested or self.config.rule_region_source
        resolved = self.config.rule_region_source or requested
        guard_result = self._guard_net_result
        guard_meta = getattr(guard_result, "metadata", None) if guard_result is not None else None
        guard_available = bool(getattr(guard_result, "available", False)) if guard_result is not None else False
        rule_region_available: bool | None
        if resolved == "guard_net":
            rule_region_available = guard_available
        elif resolved == "sam_track":
            rule_region_available = self._track_model is not None
        else:
            rule_region_available = False
        if resolved == "guard_net" and guard_meta is None:
            guard_meta = {
                "enabled": True,
                "fallback_reason": getattr(guard_result, "fallback_reason", None) if guard_result is not None else error,
            }
        rule_region_step = {
            "yaml": "yaml_roi_rule_region",
            "sam_track": "sam_track_segmentation_rule_region",
            "guard_net": "groundingdino_sam2_guard_net_rule_region",
        }.get(resolved, f"{resolved}_rule_region")
        return {
            "adapter": "sam_tracking",
            "pipeline": [
                "video_frames",
                "yolo11_object_detection",
                rule_region_step,
                "sam2_or_bbox_mask_tracking",
                "optical_flow_temporal_smoothing",
                "mask_iou_intrusion_judgment",
                "sliding_window_confirmation",
            ],
            "degraded": degraded,
            "error": error,
            "rule_region_source_requested": requested,
            "rule_region_source_resolved": resolved,
            "rule_region_available": rule_region_available,
            "guard_net": guard_meta or {"enabled": resolved == "guard_net", "fallback_reason": None},
            "object_model_path": self.config.object_model_path,
            "track_model_path": self.config.track_model_path,
            "object_model_loaded": self._object_model is not None,
            "track_model_loaded": self._track_model is not None,
            "object_model_error": self._object_model_error,
            "track_model_error": self._track_model_error,
            "sam2_status": self._sam2_error or "available",
            "sam2_enabled": self.config.sam2_enabled,
            "sam2_scan_frames": self.config.sam2_scan_frames,
            "sam2_prompt_mode": self.config.sam2_prompt_mode,
            "requested_device": self.config.device,
            "runtime_device": self._runtime_device,
            "target_labels": self.config.target_labels,
            "track_mask_labels": self.config.track_mask_labels,
            "iou_threshold": self.config.iou_threshold,
            "object_overlap_threshold": self.config.object_overlap_threshold,
            "window_size": self.config.window_size,
            "confirm_count": self.config.confirm_count,
            "use_optical_flow": self.config.use_optical_flow,
            "imgsz": self.config.imgsz,
        }

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Use CPU automatically when CUDA is requested but unavailable."""
        value = (requested or "cpu").strip().lower()
        if value in {"gpu", "cuda", "cuda:0", "0"}:
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning("SAMTracking requested CUDA but CUDA is unavailable; using CPU")
                    return "cpu"
                return "cuda:0"
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


class _SAM2VideoMemoryTracker:
    """Embedded SAM2 streaming-memory video tracker used by STEAD."""

    def __init__(
        self,
        sam2_config: str,
        sam2_checkpoint: str,
        device: str = "cuda",
        prompt_mode: str = "bounding_box",
        scan_frames: int = 30,
        new_obj_iou_thresh: float = 0.3,
    ) -> None:
        self.sam2_config = sam2_config
        self.sam2_checkpoint = sam2_checkpoint
        self.device = device
        self.prompt_mode = prompt_mode
        self.scan_frames = scan_frames
        self.new_obj_iou_thresh = new_obj_iou_thresh
        self._predictor: Any | None = None
        self._inference_state: Any | None = None
        self._propagate_gen: Any | None = None
        self._last_output_frame_idx = -1
        self._last_masks: list[Any] = []
        self._last_scores: list[float] = []
        self._last_ids: list[int] = []
        self._object_counter = 0
        self._registered_boxes: list[Any] = []
        self._has_objects = False

    def initialize_video(self, video_path: str, scan_prompts: list[dict[str, Any]] | None = None) -> None:
        """Initialize SAM2 state and register unique YOLO boxes from scan prompts."""
        predictor = self._load_model()
        self._inference_state = predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=True,
            async_loading_frames=True,
        )
        self._object_counter = 0
        self._registered_boxes = []
        self._last_output_frame_idx = -1
        self._last_masks = []
        self._last_scores = []
        self._last_ids = []
        prompts = scan_prompts or []
        for frame_idx, prompt in enumerate(prompts[: self.scan_frames]):
            boxes = prompt.get("boxes")
            if boxes is None:
                continue
            for box in boxes:
                if any(_box_iou(box, registered) > self.new_obj_iou_thresh for registered in self._registered_boxes):
                    continue
                obj_id = self._object_counter
                self._object_counter += 1
                self._registered_boxes.append(box)
                predictor.add_new_points_or_box(
                    inference_state=self._inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=box,
                )
        self._has_objects = bool(self._registered_boxes)
        self._propagate_gen = predictor.propagate_in_video(self._inference_state) if self._has_objects else None

    def track_frame(self, frame_idx: int, yolo_prompts: dict[str, Any] | None = None) -> tuple[list[Any], list[float], list[int]]:
        """Return SAM2 masks aligned to ``frame_idx`` by advancing the stream."""
        if self._inference_state is None:
            raise RuntimeError("SAM2 tracker not initialized")
        if self._propagate_gen is None:
            return [], [], []
        if frame_idx < self._last_output_frame_idx:
            return self._last_masks, self._last_scores, self._last_ids
        out_frame_idx = self._last_output_frame_idx
        out_obj_ids: Any = self._last_ids
        out_logits: Any | None = None
        while out_frame_idx < frame_idx:
            try:
                out_frame_idx, out_obj_ids, out_logits = next(self._propagate_gen)
            except StopIteration:
                return self._last_masks, self._last_scores, self._last_ids
        if out_logits is None:
            return self._last_masks, self._last_scores, self._last_ids
        try:
            import numpy as np
            try:
                import torch
            except Exception:
                torch = None  # type: ignore[assignment]

            if torch is not None and isinstance(out_logits, torch.Tensor):
                masks_tensor = (torch.sigmoid(out_logits) > 0.5).squeeze(1).detach().cpu().numpy().astype(np.uint8)
            else:
                masks_tensor = (np.asarray(out_logits) > 0.5).squeeze(1).astype(np.uint8)
            if masks_tensor.ndim == 2:
                masks_tensor = np.expand_dims(masks_tensor, 0)
            masks = [mask for mask in masks_tensor]
            ids = list(out_obj_ids) if hasattr(out_obj_ids, "__iter__") else [int(out_obj_ids)]
            self._last_output_frame_idx = int(out_frame_idx)
            self._last_masks = masks
            self._last_scores = [1.0] * len(ids)
            self._last_ids = [int(item) for item in ids]
            if self._last_output_frame_idx != frame_idx:
                logger.debug("SAM2 output aligned to frame=%s for requested frame=%s", self._last_output_frame_idx, frame_idx)
            return self._last_masks, self._last_scores, self._last_ids
        except Exception as exc:
            logger.warning("SAM2 output conversion failed at frame=%s: %s", frame_idx, exc)
            return [], [], []

    def _load_model(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from sam2.build_sam import build_sam2_video_predictor

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        config_dir, config_name = _resolve_sam2_config(self.sam2_config)
        logger.info("SAM2 config resolved: dir=%s name=%s", config_dir, config_name)
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            self._predictor = build_sam2_video_predictor(config_name, self.sam2_checkpoint, device=self.device)
        return self._predictor


class _OpticalFlowProbabilitySmoother:
    """Optical-flow guided temporal probability smoothing from the SAM2 reference pipeline."""

    def __init__(self, enabled: bool = True, entropy_epsilon: float = 1e-8, binarize_threshold: float = 0.5) -> None:
        self.enabled = enabled
        self.entropy_epsilon = entropy_epsilon
        self.binarize_threshold = binarize_threshold
        self._prev_frame: Any | None = None
        self._prev_probability: Any | None = None

    def smooth_mask(self, frame: Any, mask: Any) -> Any:
        """Smooth a binary SAM2 mask via probability fusion."""
        if not self.enabled:
            return mask
        try:
            import cv2
            import numpy as np
        except Exception:
            return mask
        probability = cv2.GaussianBlur(np.asarray(mask).astype(np.float32), (5, 5), sigmaX=1.0)
        if self._prev_frame is None or self._prev_probability is None:
            self._prev_frame = frame.copy()
            self._prev_probability = probability.copy()
            return (probability >= self.binarize_threshold).astype(np.uint8)
        try:
            prev_gray = cv2.cvtColor(self._prev_frame, cv2.COLOR_BGR2GRAY) if self._prev_frame.ndim == 3 else self._prev_frame
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = flow.shape[:2]
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            warped = cv2.remap(
                self._prev_probability,
                x_coords + flow[..., 0],
                y_coords + flow[..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            h_current = self._entropy(probability)
            h_warped = self._entropy(warped)
            alpha = h_current / (h_current + h_warped + self.entropy_epsilon)
            fused = (1.0 - alpha) * probability + alpha * warped
            self._prev_frame = frame.copy()
            self._prev_probability = fused.copy()
            return (fused >= self.binarize_threshold).astype(np.uint8)
        except Exception:
            self._prev_frame = frame.copy()
            self._prev_probability = probability.copy()
            return (probability >= self.binarize_threshold).astype(np.uint8)

    def _entropy(self, probability: Any) -> Any:
        import numpy as np

        eps = max(self.entropy_epsilon, 1e-7)
        p = np.clip(np.asarray(probability).astype(np.float64), eps, 1.0 - eps).astype(np.float32)
        return (-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)).astype(np.float32)
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


def mask_to_contours(mask: Any, max_contours: int = 8, epsilon_ratio: float = 0.003) -> list[list[list[int]]]:
    """Convert a binary mask to compact JSON-friendly contours."""
    if mask is None:
        return []
    try:
        import cv2
        import numpy as np

        binary = (np.asarray(mask) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_contours]
        encoded: list[list[list[int]]] = []
        for contour in contours:
            if len(contour) < 3 or cv2.contourArea(contour) < 16:
                continue
            epsilon = max(1.0, epsilon_ratio * cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = [[int(point[0][0]), int(point[0][1])] for point in approx]
            if len(points) >= 3:
                encoded.append(points)
        return encoded
    except Exception:
        return []


def _resolve_sam2_config(config: str) -> tuple[str, str]:
    """Resolve SAM2 config directory/name for Hydra initialization."""
    requested = Path(config)
    candidates: list[Path] = []
    repo_root = Path(__file__).resolve().parents[2]
    local_sam2_root = repo_root / "src" / "segment-anything-2" / "sam2"

    if requested.as_posix() in {"sam2_hiera_l", "sam2_hiera_l.yaml"}:
        candidates.append(local_sam2_root / "configs" / "sam2" / "sam2_hiera_l.yaml")
    if requested.exists():
        candidates.append(requested.resolve())
    cwd_candidate = (Path.cwd() / requested).resolve()
    if cwd_candidate.exists():
        candidates.append(cwd_candidate)
    candidates.extend(
        [
            repo_root / requested.name,
            repo_root / "configs" / requested.name,
            local_sam2_root / "configs" / "sam2" / requested.name,
            local_sam2_root / requested.name,
        ]
    )
    try:
        import sam2 as sam2_pkg

        sam2_root = Path(sam2_pkg.__file__).resolve().parent
        candidates.extend(
            [
                sam2_root / "configs" / "sam2" / requested.name,
                sam2_root / requested.name,
            ]
        )
    except Exception:
        pass
    for candidate in candidates:
        if candidate.exists() and _is_real_sam2_config(candidate):
            return str(candidate.parent.resolve()), candidate.stem
    fallback_parent = requested.parent if str(requested.parent) not in {"", "."} else Path(".")
    return str(fallback_parent.resolve()), requested.stem


def _is_real_sam2_config(path: Path) -> bool:
    """Return True for actual SAM2 Hydra YAML files, not pointer placeholders."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "model:" in text and "_target_" in text


def _class_allowed(label: str, class_id: int | None, allowed_labels: set[str]) -> bool:
    """Match a YOLO class by either label name or numeric id."""
    if not allowed_labels:
        return True
    candidates = {str(label)}
    if class_id is not None:
        candidates.add(str(class_id))
    return bool(candidates & allowed_labels)


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
