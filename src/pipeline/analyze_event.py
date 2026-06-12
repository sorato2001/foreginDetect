"""Command-line pipeline for STEAD event analysis."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

from src.alarm.alarm_engine import AlarmEngine
from src.config.settings import ensure_output_dir
from src.evidence.evidence_builder import build_object_tracks, empty_evidence
from src.evidence.evidence_schema import ROIRuleTrigger
from src.evidence.serializers import save_json, save_model
from src.evidence.window_builder import WindowBuilder
from src.perception.detector import Detection
from src.perception.sam_tracking_adapter import SAMTrackingAdapter, SAMTrackingConfig, SAMTrackingResult
from src.perception.simple_iou_tracker import SimpleIOUTracker
from src.perception.yolo_detector import YoloDetector
from src.rules.rule_engine import RuleEngine
from src.vlm.mock_provider import MockVLMProvider
from src.vlm.qwen_provider import QwenProvider
from src.vlm.vlm_schema import VLMReview
from src.visualization.pipeline_visualizer import save_pipeline_visualization

logger = logging.getLogger("stead.pipeline")


def _parse_bool(value: str | bool) -> bool:
    """Parse CLI boolean values."""
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _fake_tracks() -> dict[int, list[Detection]]:
    """Return deterministic tracks for smoke tests and offline demos."""
    return {
        1: [
            Detection("person", 0.85, [120, 120, 170, 220], 0, 0.0),
            Detection("person", 0.88, [180, 150, 230, 250], 1, 2.0),
            Detection("person", 0.90, [260, 180, 310, 280], 2, 4.0),
        ]
    }


def _configure_pipeline_logging(output_dir: Path, log_level: str = "INFO") -> str:
    """Configure console and per-run file logging without leaking secrets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        if getattr(handler, "_stead_pipeline_file", False):
            root.removeHandler(handler)
            handler.close()

    if not any(getattr(handler, "_stead_pipeline_console", False) for handler in root.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        console._stead_pipeline_console = True  # type: ignore[attr-defined]
        root.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    file_handler._stead_pipeline_file = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)
    return str(log_path)


def _detect_video(
    video_path: str,
    mock_detections: bool = False,
    max_analysis_frames: int | None = 900,
) -> tuple[dict[int, list[Detection]], float, float]:
    """Detect and track objects in a video with graceful fallback."""
    if mock_detections:
        logger.info("STEP 02 detector/tracker: using deterministic mock detections")
        return _fake_tracks(), 25.0, 6.0
    try:
        import cv2
    except Exception as exc:
        logger.warning("STEP 02 detector/tracker: OpenCV unavailable, returning empty tracks: %s", type(exc).__name__)
        return {}, 0.0, 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("STEP 02 detector/tracker: cannot open video=%s", video_path)
        return {}, 0.0, 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    detector = YoloDetector()
    tracker = SimpleIOUTracker()
    sample_every = max(1, int(fps))
    frame_index = 0
    while True:
        if max_analysis_frames and frame_index >= max_analysis_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_every == 0:
            timestamp = frame_index / fps if fps > 0 else 0.0
            tracker.update(detector.detect_frame(frame, frame_index, timestamp))
        frame_index += 1
    cap.release()
    duration = frame_index / fps if fps > 0 else 0.0
    if not max_analysis_frames and fps > 0 and total:
        duration = total / fps
    return tracker.tracks, fps, duration


def _detect_video_with_sam_tracking(
    video_path: str,
    config: SAMTrackingConfig,
    max_analysis_frames: int | None = 900,
) -> SAMTrackingResult:
    """Run the optional SAMTracking tracker adapter."""
    adapter = SAMTrackingAdapter(config)
    return adapter.analyze_video(video_path, max_frames=max_analysis_frames)


def run_pipeline(
    video_path: str,
    camera_id: str,
    rules_path: str,
    output_dir: str,
    vlm_provider: str = "mock",
    event_id: str | None = None,
    mock_detections: bool = False,
    vlm_timeout: float = 20.0,
    vlm_max_retries: int = 0,
    vlm_fallback_on_error: bool = True,
    save_visualization: bool = True,
    max_analysis_frames: int | None = 900,
    visualization_max_frames: int | None = 300,
    log_level: str = "INFO",
    tracker: str = "simple_iou",
    sam_object_model: str | None = None,
    sam_track_model: str | None = None,
    sam2_config: str | None = None,
    sam2_checkpoint: str | None = None,
    sam_device: str = "cuda",
    sam_iou_threshold: float = 0.10,
    sam_object_overlap_threshold: float = 0.15,
    sam_window_size: int = 5,
    sam_confirm_count: int = 3,
    sam_use_optical_flow: bool = True,
    sam2_enabled: bool = True,
    sam2_scan_frames: int = 30,
    sam2_prompt_mode: str = "bounding_box",
    sam_sample_every: int = 15,
    sam_track_mask_interval: int = 30,
    sam_imgsz: int = 640,
    sam_progress_interval: int = 10,
) -> dict:
    """Run STEAD event analysis and write JSON artifacts."""
    event_id = event_id or f"event_{uuid.uuid4().hex[:8]}"
    out = ensure_output_dir(output_dir)
    log_path = _configure_pipeline_logging(out, log_level=log_level)
    pipeline_start = time.perf_counter()
    logger.info(
        "STEP 01 start: event_id=%s camera_id=%s video=%s output=%s vlm_provider=%s",
        event_id,
        camera_id,
        video_path,
        out,
        vlm_provider,
    )
    logger.info(
        "STEP 01 config: rules=%s tracker=%s mock_detections=%s max_analysis_frames=%s save_visualization=%s visualization_max_frames=%s",
        rules_path,
        tracker,
        mock_detections,
        max_analysis_frames,
        save_visualization,
        visualization_max_frames,
    )

    step_start = time.perf_counter()
    logger.info("STEP 02 detector/tracker: begin tracker=%s", tracker)
    sam_tracking_result: SAMTrackingResult | None = None
    if tracker == "sam_tracking" and not mock_detections:
        sam_config = SAMTrackingConfig(
            object_model_path=sam_object_model or SAMTrackingConfig().object_model_path,
            track_model_path=sam_track_model or SAMTrackingConfig().track_model_path,
            sam2_config=sam2_config,
            sam2_checkpoint=sam2_checkpoint,
            device=sam_device,
            iou_threshold=sam_iou_threshold,
            object_overlap_threshold=sam_object_overlap_threshold,
            window_size=sam_window_size,
            confirm_count=sam_confirm_count,
            use_optical_flow=sam_use_optical_flow,
            sam2_enabled=sam2_enabled,
            sam2_scan_frames=sam2_scan_frames,
            sam2_prompt_mode=sam2_prompt_mode,
            sample_every=sam_sample_every,
            track_mask_interval=sam_track_mask_interval,
            imgsz=sam_imgsz,
            progress_interval=sam_progress_interval,
        )
        sam_tracking_result = _detect_video_with_sam_tracking(
            video_path,
            sam_config,
            max_analysis_frames=max_analysis_frames,
        )
        tracks, fps, duration = sam_tracking_result.tracks, sam_tracking_result.fps, sam_tracking_result.duration
    else:
        if tracker == "sam_tracking" and mock_detections:
            logger.info("STEP 02 detector/tracker: mock_detections requested; using deterministic tracks instead of SAMTracking")
        tracks, fps, duration = _detect_video(video_path, mock_detections=mock_detections, max_analysis_frames=max_analysis_frames)
    detection_count = sum(len(history) for history in tracks.values())
    logger.info(
        "STEP 02 detector/tracker: done tracker=%s tracks=%s detections=%s fps=%.3f duration=%.3fs elapsed=%.3fs",
        tracker,
        len(tracks),
        detection_count,
        fps,
        duration,
        time.perf_counter() - step_start,
    )

    step_start = time.perf_counter()
    logger.info("STEP 03 evidence: build object tracks")
    evidence = empty_evidence(event_id, camera_id, video_path, duration=duration, fps=fps or None)
    evidence.objects = build_object_tracks(tracks)
    logger.info(
        "STEP 03 evidence: done objects=%s elapsed=%.3fs",
        len(evidence.objects),
        time.perf_counter() - step_start,
    )

    step_start = time.perf_counter()
    logger.info("STEP 04 ROI/rules: load rules and evaluate")
    rule_engine = RuleEngine.from_yaml(rules_path)
    sam_rule = _sam_tracking_mask_iou_rule(sam_tracking_result)
    if sam_rule is not None:
        evidence.roi_rules = [sam_rule]
        evidence.metadata["rule_source"] = "sam_tracking_mask_iou"
        logger.info("STEP 04 ROI/rules: using SAMTracking mask IoU rule because track mask is available")
    else:
        evidence.roi_rules = rule_engine.evaluate(evidence)
        evidence.metadata["rule_source"] = "config_rules"
        if tracker == "sam_tracking":
            logger.info("STEP 04 ROI/rules: SAMTracking track mask unavailable; fallback to config rules")
    triggered_rules = [rule.rule_id for rule in evidence.roi_rules if rule.triggered]
    logger.info(
        "STEP 04 ROI/rules: done rules=%s triggered=%s elapsed=%.3fs",
        len(evidence.roi_rules),
        triggered_rules,
        time.perf_counter() - step_start,
    )

    step_start = time.perf_counter()
    logger.info("STEP 05 windows: build temporal windows")
    evidence.windows = WindowBuilder().build(evidence)
    evidence.metadata.update({"mock_detections": mock_detections, "stead_version": "stead_v1", "tracker": tracker})
    sam_tracking_artifact: str | None = None
    if sam_tracking_result is not None:
        sam_tracking_artifact = str(out / "sam_tracking_result.json")
        save_json(sam_tracking_result.to_json_dict(), sam_tracking_artifact)
        evidence.metadata["sam_tracking"] = {
            "artifact": sam_tracking_artifact,
            "degraded": sam_tracking_result.metadata.get("degraded"),
            "processed_frames": sam_tracking_result.metadata.get("processed_frames"),
            "intrusion_events": len(sam_tracking_result.intrusion_events),
            "track_mask_seen": sam_tracking_result.metadata.get("track_mask_seen"),
            "detections_seen": sam_tracking_result.metadata.get("detections_seen"),
            "summary": _sam_tracking_prompt_summary(sam_tracking_result),
        }
        logger.info(
            "STEP 05 windows: SAMTracking artifact=%s degraded=%s intrusion_events=%s",
            sam_tracking_artifact,
            sam_tracking_result.metadata.get("degraded"),
            len(sam_tracking_result.intrusion_events),
        )
    logger.info("STEP 05 windows: done windows=%s elapsed=%.3fs", len(evidence.windows), time.perf_counter() - step_start)

    visualization_artifacts = {}
    if save_visualization:
        step_start = time.perf_counter()
        logger.info("STEP 06 visualization: render detector/tracker/rule outputs")
        visualization_artifacts = save_pipeline_visualization(
            evidence,
            rules_path,
            str(out),
            video_path,
            max_video_frames=visualization_max_frames,
        )
        evidence.metadata["visualization"] = visualization_artifacts
        logger.info(
            "STEP 06 visualization: done overview=%s annotated_video=%s evidence_animation=%s summary=%s elapsed=%.3fs",
            visualization_artifacts.get("overview_image"),
            visualization_artifacts.get("annotated_video"),
            visualization_artifacts.get("evidence_animation"),
            visualization_artifacts.get("summary_json"),
            time.perf_counter() - step_start,
        )
    else:
        logger.info("STEP 06 visualization: skipped")

    step_start = time.perf_counter()
    logger.info(
        "STEP 07 VLM: begin provider=%s timeout=%.1fs max_retries=%s fallback_on_error=%s",
        vlm_provider,
        vlm_timeout,
        vlm_max_retries,
        vlm_fallback_on_error,
    )
    provider = (
        QwenProvider(
            timeout_sec=vlm_timeout,
            read_timeout_seconds=vlm_timeout,
            max_retries=vlm_max_retries,
            fallback_on_error=vlm_fallback_on_error,
            artifact_dir=str(out),
        )
        if vlm_provider == "qwen"
        else MockVLMProvider()
    )
    try:
        review = provider.review(evidence)
    except Exception as exc:
        logger.exception("STEP 07 VLM: provider raised unexpectedly, using pipeline fallback")
        review = _pipeline_fallback_review(exc, provider_name=vlm_provider)
        if vlm_provider == "qwen":
            save_json(
                {
                    "provider": "qwen",
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                    "fallback": True,
                    "pipeline_fallback": True,
                },
                str(out / "qwen_error.json"),
            )
    logger.info(
        "STEP 07 VLM: done level=%s confidence=%.3f success=%s fallback=%s elapsed=%.3fs",
        review.alarm_level_suggestion,
        review.confidence,
        review.metadata.get("success"),
        review.metadata.get("fallback"),
        time.perf_counter() - step_start,
    )

    step_start = time.perf_counter()
    logger.info("STEP 08 alarm: fuse rule and VLM scores")
    alarm = AlarmEngine().fuse(evidence, review)
    logger.info(
        "STEP 08 alarm: done final_level=%s final_score=%.3f is_alarm=%s elapsed=%.3fs",
        alarm.final_level,
        alarm.final_score,
        alarm.is_alarm,
        time.perf_counter() - step_start,
    )

    step_start = time.perf_counter()
    logger.info("STEP 09 artifacts: save JSON outputs")
    save_model(evidence, str(out / "event_evidence.json"))
    save_model(review, str(out / "vlm_review.json"))
    save_model(alarm, str(out / "alarm_result.json"))
    logger.info(
        "STEP 09 artifacts: done evidence=%s review=%s alarm=%s log=%s elapsed=%.3fs",
        out / "event_evidence.json",
        out / "vlm_review.json",
        out / "alarm_result.json",
        log_path,
        time.perf_counter() - step_start,
    )
    logger.info("STEP 10 complete: event_id=%s total_elapsed=%.3fs", event_id, time.perf_counter() - pipeline_start)
    return {
        "event_id": event_id,
        "output_dir": str(out),
        "event_evidence": str(out / "event_evidence.json"),
        "vlm_review": str(out / "vlm_review.json"),
        "alarm_result": str(out / "alarm_result.json"),
        "visualization": visualization_artifacts,
        "pipeline_log": log_path,
        "sam_tracking_result": sam_tracking_artifact,
        "final_level": alarm.final_level,
        "is_alarm": alarm.is_alarm,
    }


def _sam_tracking_mask_iou_rule(sam_tracking_result: SAMTrackingResult | None) -> ROIRuleTrigger | None:
    """Build a STEP 04 rule result from SAMTracking mask-IoU intrusion judgment."""
    if sam_tracking_result is None or not sam_tracking_result.metadata.get("track_mask_seen"):
        return None
    alarm_frames = [frame for frame in sam_tracking_result.frames if frame.alarm]
    suspicious_frames = [frame for frame in sam_tracking_result.frames if frame.suspicious]
    evidence_tracks = sorted(
        {
            track_id
            for frame in suspicious_frames or alarm_frames
            for det in frame.detections
            for track_id, history in sam_tracking_result.tracks.items()
            if any(item.frame_index == frame.frame_index and item.label == det.get("label") for item in history)
        }
    )
    triggered = bool(alarm_frames or suspicious_frames)
    trigger_frames = alarm_frames or suspicious_frames
    trigger_time = [trigger_frames[0].timestamp, trigger_frames[-1].timestamp] if trigger_frames else None
    peak_iou = max((frame.max_iou for frame in sam_tracking_result.frames), default=0.0)
    severity = "high" if alarm_frames else "medium" if suspicious_frames else "none"
    return ROIRuleTrigger(
        rule_id="sam_mask_iou_intrusion",
        rule_type="mask_iou_intrusion",
        roi_id="segmented_railway_track",
        triggered=triggered,
        trigger_time=trigger_time,
        evidence_tracks=evidence_tracks,
        severity_hint=severity,
    )


def _sam_tracking_prompt_summary(sam_tracking_result: SAMTrackingResult) -> dict:
    """Build a compact SAMTracking summary for VLM review."""
    frames = sam_tracking_result.frames
    suspicious = [frame for frame in frames if frame.suspicious]
    alarms = [frame for frame in frames if frame.alarm]
    peak = max(frames, key=lambda frame: max(frame.max_iou, frame.max_object_overlap), default=None)
    sample_frames = alarms[:3] or suspicious[:3] or ([peak] if peak is not None else [])
    return {
        "rule_type": "mask_iou_intrusion",
        "track_mask_seen": sam_tracking_result.metadata.get("track_mask_seen"),
        "detections_seen": sam_tracking_result.metadata.get("detections_seen"),
        "processed_frames": sam_tracking_result.metadata.get("processed_frames"),
        "sample_every": sam_tracking_result.metadata.get("sample_every"),
        "track_mask_interval": sam_tracking_result.metadata.get("track_mask_interval"),
        "iou_threshold": sam_tracking_result.metadata.get("iou_threshold"),
        "object_overlap_threshold": sam_tracking_result.metadata.get("object_overlap_threshold"),
        "window_size": sam_tracking_result.metadata.get("window_size"),
        "confirm_count": sam_tracking_result.metadata.get("confirm_count"),
        "max_iou": round(peak.max_iou, 4) if peak is not None else 0.0,
        "max_object_overlap": round(peak.max_object_overlap, 4) if peak is not None else 0.0,
        "peak_frame": peak.frame_index if peak is not None else None,
        "suspicious_frame_count": len(suspicious),
        "alarm_frame_count": len(alarms),
        "intrusion_event_count": len(sam_tracking_result.intrusion_events),
        "sample_frames": [
            {
                "frame_index": frame.frame_index,
                "timestamp": round(frame.timestamp, 3),
                "max_iou": round(frame.max_iou, 4),
                "max_object_overlap": round(frame.max_object_overlap, 4),
                "suspicious": frame.suspicious,
                "alarm": frame.alarm,
                "window_count": frame.window_count,
                "detections": [
                    {
                        "label": det.get("label"),
                        "confidence": round(float(det.get("confidence", 0.0)), 3),
                        "bbox": det.get("bbox"),
                    }
                    for det in frame.detections[:5]
                ],
            }
            for frame in sample_frames
            if frame is not None
        ],
    }


def _pipeline_fallback_review(exc: Exception, provider_name: str) -> VLMReview:
    """Last-resort fallback if a provider unexpectedly raises."""
    error_type = type(exc).__name__
    return VLMReview(
        is_anomaly=False,
        event_type="none",
        alarm_level_suggestion="none",
        confidence=0.0,
        evidence_time=[],
        evidence_tracks=[],
        matched_rules=[],
        reason=f"{provider_name} review failed in pipeline: {error_type}. Fallback review generated.",
        possible_false_alarm=True,
        recommended_action="manual review recommended",
        metadata={
            "provider": provider_name,
            "success": False,
            "error_type": error_type,
            "error_message": str(exc)[:500],
            "fallback": True,
            "pipeline_fallback": True,
        },
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze one surveillance event with STEAD.")
    parser.add_argument("--input-type", choices=["video", "image"], default="video")
    parser.add_argument("--video", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--rules", default="configs/rules.example.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vlm-provider", choices=["mock", "qwen"], default="mock")
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--mock-detections", action="store_true")
    parser.add_argument("--vlm-timeout", type=float, default=20.0)
    parser.add_argument("--vlm-max-retries", type=int, default=0)
    parser.add_argument("--vlm-fallback-on-error", default="true")
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--max-analysis-frames", type=int, default=900, help="Max video frames to scan; 0 means full video")
    parser.add_argument("--visualization-max-frames", type=int, default=300, help="Max annotated-video frames; 0 means full video")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--tracker", "--track", dest="tracker", choices=["simple_iou", "sam_tracking"], default="simple_iou")
    parser.add_argument("--sam-object-model", default=None, help="YOLO11 object model path for SAMTracking")
    parser.add_argument("--sam-track-model", default=None, help="Rail/track segmentation best.pt path for SAMTracking")
    parser.add_argument("--sam2-config", default=None, help="SAM2 config path for SAMTracking")
    parser.add_argument("--sam2-checkpoint", default=None, help="SAM2 checkpoint path for SAMTracking")
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--sam-iou-threshold", type=float, default=0.10)
    parser.add_argument("--sam-object-overlap-threshold", type=float, default=0.15)
    parser.add_argument("--sam-window-size", type=int, default=5)
    parser.add_argument("--sam-confirm-count", type=int, default=3)
    parser.add_argument("--sam-use-optical-flow", default="true")
    parser.add_argument("--sam2-enabled", default="true")
    parser.add_argument("--sam2-scan-frames", type=int, default=30)
    parser.add_argument("--sam2-prompt-mode", default="bounding_box", choices=["bounding_box", "center_point", "centroid"])
    parser.add_argument("--sam-sample-every", type=int, default=15, help="Run object detection every N frames in SAMTracking")
    parser.add_argument("--sam-track-mask-interval", type=int, default=30, help="Run railway mask segmentation every N frames in SAMTracking")
    parser.add_argument("--sam-imgsz", type=int, default=640, help="YOLO inference image size for SAMTracking")
    parser.add_argument("--sam-progress-interval", type=int, default=10, help="Log SAMTracking progress every N processed frames")
    args = parser.parse_args()
    if args.input_type == "image":
        if not args.image:
            parser.error("--image is required when --input-type image")
        from src.pipeline.analyze_image import run_image_pipeline

        result = run_image_pipeline(
            image_path=args.image,
            camera_id=args.camera_id,
            rules_path=args.rules,
            output_dir=args.output,
            vlm_provider=args.vlm_provider,
            event_id=args.event_id,
            mock_detections=args.mock_detections,
            vlm_timeout=args.vlm_timeout,
            vlm_max_retries=args.vlm_max_retries,
            vlm_fallback_on_error=_parse_bool(args.vlm_fallback_on_error),
            save_visualization=not args.no_visualization,
            log_level=args.log_level,
        )
    else:
        if not args.video:
            parser.error("--video is required when --input-type video")
        result = run_pipeline(
            video_path=args.video,
            camera_id=args.camera_id,
            rules_path=args.rules,
            output_dir=args.output,
            vlm_provider=args.vlm_provider,
            event_id=args.event_id,
            mock_detections=args.mock_detections,
            vlm_timeout=args.vlm_timeout,
            vlm_max_retries=args.vlm_max_retries,
            vlm_fallback_on_error=_parse_bool(args.vlm_fallback_on_error),
            save_visualization=not args.no_visualization,
            max_analysis_frames=args.max_analysis_frames or None,
            visualization_max_frames=args.visualization_max_frames or None,
            log_level=args.log_level,
            tracker=args.tracker,
            sam_object_model=args.sam_object_model,
            sam_track_model=args.sam_track_model,
            sam2_config=args.sam2_config,
            sam2_checkpoint=args.sam2_checkpoint,
            sam_device=args.sam_device,
            sam_iou_threshold=args.sam_iou_threshold,
            sam_object_overlap_threshold=args.sam_object_overlap_threshold,
            sam_window_size=args.sam_window_size,
            sam_confirm_count=args.sam_confirm_count,
            sam_use_optical_flow=_parse_bool(args.sam_use_optical_flow),
            sam2_enabled=_parse_bool(args.sam2_enabled),
            sam2_scan_frames=args.sam2_scan_frames,
            sam2_prompt_mode=args.sam2_prompt_mode,
            sam_sample_every=args.sam_sample_every,
            sam_track_mask_interval=args.sam_track_mask_interval,
            sam_imgsz=args.sam_imgsz,
            sam_progress_interval=args.sam_progress_interval,
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
