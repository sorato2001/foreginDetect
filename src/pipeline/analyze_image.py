"""Image analysis pipeline for STEAD.

This module is intentionally separate from ``analyze_event.py`` so image input
can be added without disturbing the existing video pipeline.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from src.alarm.alarm_engine import AlarmEngine
from src.config.settings import ensure_output_dir
from src.evidence.evidence_builder import build_object_tracks, empty_evidence
from src.evidence.evidence_schema import KeyframeInfo
from src.evidence.serializers import save_json, save_model
from src.evidence.window_builder import WindowBuilder
from src.perception.detector import Detection
from src.perception.sam_tracking_adapter import SAMTrackingAdapter, SAMTrackingConfig, SAMTrackingResult
from src.perception.yolo_detector import YoloDetector
from src.pipeline.analyze_event import (
    _configure_pipeline_logging,
    _copy_batch_visualizations,
    _sam_tracking_mask_iou_rule,
    _sam_tracking_prompt_summary,
    _sam_tracking_rule_source,
    logger,
)
from src.rules.rule_engine import RuleEngine
from src.vlm.pipeline_runner import run_vlm_review
from src.vlm.provider_factory import resolve_vlm_provider
from src.visualization.pipeline_visualizer import save_image_visualization

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _fake_image_tracks() -> dict[int, list[Detection]]:
    """Return deterministic single-image detections for offline demos."""
    return {
        1: [Detection("person", 0.88, [120, 120, 180, 260], frame_index=0, timestamp=0.0)],
    }


def _detect_image(image_path: str, mock_detections: bool = False) -> dict[int, list[Detection]]:
    """Detect objects on a single image and return one-frame track histories."""
    if mock_detections:
        logger.info("STEP 02 detector/tracker: using deterministic mock image detections")
        return _fake_image_tracks()
    try:
        import cv2
    except Exception as exc:
        logger.warning("STEP 02 detector/tracker: OpenCV unavailable for image input: %s", type(exc).__name__)
        return {}

    frame = cv2.imread(image_path)
    if frame is None:
        logger.warning("STEP 02 detector/tracker: cannot read image=%s", image_path)
        return {}
    detector = YoloDetector()
    detections = detector.detect_frame(frame, frame_index=0, timestamp=0.0)
    return {idx + 1: [detection] for idx, detection in enumerate(detections)}


def _detect_image_with_sam_tracking(image_path: str, config: SAMTrackingConfig) -> SAMTrackingResult:
    """Run SAMTracking's single-image railway intrusion logic."""
    adapter = SAMTrackingAdapter(config)
    return adapter.analyze_image(image_path)


def collect_image_paths(image_path: str | Path, recursive: bool = False, limit: int | None = None) -> list[Path]:
    """Collect one image path or all supported images in a directory."""
    root = Path(image_path)
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_SUFFIXES else []
    if not root.is_dir():
        return []
    candidates = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if limit is not None and limit > 0:
        return paths[:limit]
    return paths


def _batch_item_dir(output_dir: Path, source_root: Path, image_path: Path) -> Path:
    """Return a stable per-image output directory."""
    if source_root.is_dir():
        try:
            relative = image_path.relative_to(source_root).with_suffix("")
        except ValueError:
            relative = Path(image_path.stem)
    else:
        relative = Path(image_path.stem)
    safe_parts = [part.replace(" ", "_") for part in relative.parts]
    return output_dir.joinpath(*safe_parts)


def run_image_pipeline(
    image_path: str,
    camera_id: str,
    rules_path: str,
    output_dir: str,
    vlm_provider: str = "mock",
    vlm_mode: str = "web",
    vlm_local_endpoint: str = "http://localhost:8082/v1/chat/completions",
    vlm_local_model: str = "gemma-4-26B",
    vlm_local_max_images: int = 1,
    event_id: str | None = None,
    mock_detections: bool = False,
    vlm_timeout: float = 20.0,
    vlm_max_retries: int = 0,
    vlm_fallback_on_error: bool = True,
    save_visualization: bool = True,
    log_level: str = "INFO",
    tracker: str = "simple_iou",
    sam_object_model: str | None = None,
    sam_track_model: str | None = None,
    sam_track_labels: list[str] | None = None,
    sam2_config: str | None = None,
    sam2_checkpoint: str | None = None,
    sam2_enabled: bool = True,
    sam_device: str = "cuda",
    sam_iou_threshold: float = 0.10,
    sam_object_overlap_threshold: float = 0.15,
    sam_imgsz: int = 640,
    rule_region_source: str = "auto",
) -> dict[str, Any]:
    """Run STEAD analysis on a single image and write standard artifacts."""
    event_id = event_id or f"image_{uuid.uuid4().hex[:8]}"
    out = ensure_output_dir(output_dir)
    log_path = _configure_pipeline_logging(out, log_level=log_level)
    pipeline_start = time.perf_counter()
    effective_vlm_provider = resolve_vlm_provider(vlm_provider, vlm_mode)
    logger.info(
        "STEP 01 start: event_id=%s camera_id=%s image=%s output=%s vlm_provider=%s vlm_mode=%s",
        event_id,
        camera_id,
        image_path,
        out,
        effective_vlm_provider,
        vlm_mode,
    )
    logger.info(
        "STEP 01 config: rules=%s tracker=%s mock_detections=%s save_visualization=%s",
        rules_path,
        tracker,
        mock_detections,
        save_visualization,
    )

    step_start = time.perf_counter()
    logger.info("STEP 02 detector/tracker: begin image detection tracker=%s", tracker)
    sam_tracking_result: SAMTrackingResult | None = None
    if tracker == "sam_tracking" and not mock_detections:
        from src.pipeline.analyze_event import _resolve_rule_region_source

        resolved_rule_region_source = _resolve_rule_region_source(rule_region_source, tracker, sam_track_model)
        sam_config = SAMTrackingConfig(
            object_model_path=sam_object_model or SAMTrackingConfig().object_model_path,
            track_model_path=sam_track_model or SAMTrackingConfig().track_model_path,
            rule_region_source=resolved_rule_region_source,
            rule_region_source_requested=rule_region_source,
            track_mask_labels=sam_track_labels or [],
            device=sam_device,
            iou_threshold=sam_iou_threshold,
            object_overlap_threshold=sam_object_overlap_threshold,
            window_size=1,
            confirm_count=1,
            sam2_config=sam2_config,
            sam2_checkpoint=sam2_checkpoint,
            sam2_enabled=sam2_enabled,
            sample_every=1,
            track_mask_interval=1,
            imgsz=sam_imgsz,
            output_dir=str(out),
        )
        sam_tracking_result = _detect_image_with_sam_tracking(image_path, sam_config)
        tracks = sam_tracking_result.tracks
    else:
        if tracker == "sam_tracking" and mock_detections:
            logger.info("STEP 02 detector/tracker: mock_detections requested; using deterministic image tracks instead of SAMTracking")
        tracks = _detect_image(image_path, mock_detections=mock_detections)
    detection_count = sum(len(history) for history in tracks.values())
    logger.info(
        "STEP 02 detector/tracker: done tracker=%s tracks=%s detections=%s elapsed=%.3fs",
        tracker,
        len(tracks),
        detection_count,
        time.perf_counter() - step_start,
    )

    step_start = time.perf_counter()
    logger.info("STEP 03 evidence: build single-image evidence")
    evidence = empty_evidence(event_id, camera_id, image_path, duration=1.0, fps=None)
    evidence.objects = build_object_tracks(tracks)
    evidence.keyframes = [
        KeyframeInfo(
            timestamp=0.0,
            frame_path=image_path,
            reason="input_image",
            boxes=[
                {
                    "track_id": track_id,
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "bbox": detection.bbox,
                }
                for track_id, history in tracks.items()
                for detection in history
            ],
        )
    ]
    evidence.metadata.update({"input_type": "image", "mock_detections": mock_detections, "stead_version": "stead_v1"})
    logger.info("STEP 03 evidence: done objects=%s elapsed=%.3fs", len(evidence.objects), time.perf_counter() - step_start)

    step_start = time.perf_counter()
    logger.info("STEP 04 ROI/rules: load rules and evaluate")
    rule_engine = RuleEngine.from_yaml(rules_path)
    sam_rule = _sam_tracking_mask_iou_rule(sam_tracking_result)
    if sam_rule is not None:
        evidence.roi_rules = [sam_rule]
        evidence.metadata["rule_source"] = _sam_tracking_rule_source(sam_tracking_result)
        logger.info(
            "STEP 04 ROI/rules: using %s mask IoU rule because rule-region mask is available",
            sam_tracking_result.metadata.get("rule_region_source_resolved"),
        )
    else:
        evidence.roi_rules = rule_engine.evaluate(evidence)
        evidence.metadata["rule_source"] = "config_rules"
        if tracker == "sam_tracking":
            source = sam_tracking_result.metadata.get("rule_region_source_resolved") if sam_tracking_result else None
            logger.info("STEP 04 ROI/rules: %s rule-region mask unavailable; fallback to config rules", source or "SAMTracking")
    triggered_rules = [rule.rule_id for rule in evidence.roi_rules if rule.triggered]
    logger.info(
        "STEP 04 ROI/rules: done rules=%s triggered=%s elapsed=%.3fs",
        len(evidence.roi_rules),
        triggered_rules,
        time.perf_counter() - step_start,
    )

    step_start = time.perf_counter()
    logger.info("STEP 05 windows: build image window")
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
            "rule_region_source": sam_tracking_result.metadata.get("rule_region_source_resolved"),
            "rule_region_available": sam_tracking_result.metadata.get("rule_region_available"),
            "detections_seen": sam_tracking_result.metadata.get("detections_seen"),
            "summary": _sam_tracking_prompt_summary(sam_tracking_result),
        }
        logger.info(
            "STEP 05 windows: SAMTracking artifact=%s degraded=%s intrusion_events=%s",
            sam_tracking_artifact,
            sam_tracking_result.metadata.get("degraded"),
            len(sam_tracking_result.intrusion_events),
        )
    evidence.windows = WindowBuilder(window_size=1.0, stride=1.0).build(evidence)
    logger.info("STEP 05 windows: done windows=%s elapsed=%.3fs", len(evidence.windows), time.perf_counter() - step_start)

    visualization_artifacts = {}
    if save_visualization:
        step_start = time.perf_counter()
        logger.info("STEP 06 visualization: render image detector/tracker/rule outputs")
        visualization_artifacts = save_image_visualization(evidence, rules_path, str(out), image_path=image_path)
        evidence.metadata["visualization"] = visualization_artifacts
        logger.info(
            "STEP 06 visualization: done annotated_image=%s summary=%s elapsed=%.3fs",
            visualization_artifacts.get("annotated_image"),
            visualization_artifacts.get("summary_json"),
            time.perf_counter() - step_start,
        )
    else:
        logger.info("STEP 06 visualization: skipped")

    step_start = time.perf_counter()
    logger.info("STEP 07 VLM: begin provider=%s mode=%s timeout=%.1fs max_retries=%s", effective_vlm_provider, vlm_mode, vlm_timeout, vlm_max_retries)
    review, effective_vlm_provider = run_vlm_review(
        evidence=evidence,
        output_dir=out,
        vlm_provider=effective_vlm_provider,
        vlm_mode=vlm_mode,
        timeout_sec=vlm_timeout,
        max_retries=vlm_max_retries,
        fallback_on_error=vlm_fallback_on_error,
        local_endpoint=vlm_local_endpoint,
        local_model=vlm_local_model,
        local_max_images=vlm_local_max_images,
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
        "input_type": "image",
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


def run_image_batch_pipeline(
    image_path: str,
    camera_id: str,
    rules_path: str,
    output_dir: str,
    vlm_provider: str = "mock",
    vlm_mode: str = "web",
    vlm_local_endpoint: str = "http://localhost:8082/v1/chat/completions",
    vlm_local_model: str = "gemma-4-26B",
    vlm_local_max_images: int = 1,
    mock_detections: bool = False,
    vlm_timeout: float = 20.0,
    vlm_max_retries: int = 0,
    vlm_fallback_on_error: bool = True,
    save_visualization: bool = True,
    log_level: str = "INFO",
    tracker: str = "simple_iou",
    sam_object_model: str | None = None,
    sam_track_model: str | None = None,
    sam_track_labels: list[str] | None = None,
    sam2_config: str | None = None,
    sam2_checkpoint: str | None = None,
    sam2_enabled: bool = True,
    sam_device: str = "cuda",
    sam_iou_threshold: float = 0.10,
    sam_object_overlap_threshold: float = 0.15,
    sam_imgsz: int = 640,
    rule_region_source: str = "auto",
    recursive: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run image analysis for a file or directory and write a batch summary."""
    source = Path(image_path)
    out = ensure_output_dir(output_dir)
    image_paths = collect_image_paths(source, recursive=recursive, limit=limit)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    batch_visualizations: list[dict[str, str]] = []
    batch_vis_dir = out / "batch_visualizations"

    for index, item in enumerate(image_paths, start=1):
        item_output = _batch_item_dir(out, source, item)
        event_id = f"image_{item.stem}_{index:04d}"
        try:
            result = run_image_pipeline(
                image_path=str(item),
                camera_id=camera_id,
                rules_path=rules_path,
                output_dir=str(item_output),
                vlm_provider=vlm_provider,
                vlm_mode=vlm_mode,
                vlm_local_endpoint=vlm_local_endpoint,
                vlm_local_model=vlm_local_model,
                vlm_local_max_images=vlm_local_max_images,
                event_id=event_id,
                mock_detections=mock_detections,
                vlm_timeout=vlm_timeout,
                vlm_max_retries=vlm_max_retries,
                vlm_fallback_on_error=vlm_fallback_on_error,
                save_visualization=save_visualization,
                log_level=log_level,
                tracker=tracker,
                sam_object_model=sam_object_model,
                sam_track_model=sam_track_model,
                sam_track_labels=sam_track_labels,
                sam2_config=sam2_config,
                sam2_checkpoint=sam2_checkpoint,
                sam2_enabled=sam2_enabled,
                sam_device=sam_device,
                sam_iou_threshold=sam_iou_threshold,
                sam_object_overlap_threshold=sam_object_overlap_threshold,
                sam_imgsz=sam_imgsz,
                rule_region_source=rule_region_source,
            )
            result["source_image"] = str(item)
            copied_visualizations = _copy_batch_visualizations(result, batch_vis_dir, f"{index:04d}_{item.stem}")
            result["batch_visualizations"] = copied_visualizations
            batch_visualizations.extend(copied_visualizations)
            results.append(result)
        except Exception as exc:
            logger.exception("Batch image analysis failed: image=%s", item)
            failures.append({"image": str(item), "error_type": type(exc).__name__, "error_message": str(exc)[:500]})

    summary = {
        "input_type": "image_batch",
        "source": str(source),
        "output_dir": str(out),
        "recursive": recursive,
        "total": len(image_paths),
        "succeeded": len(results),
        "failed": len(failures),
        "alarm_count": sum(1 for item in results if item.get("is_alarm")),
        "batch_visualization_dir": str(batch_vis_dir),
        "batch_visualizations": batch_visualizations,
        "results": results,
        "failures": failures,
    }
    summary_path = out / "batch_summary.json"
    save_json(summary, str(summary_path))
    summary["summary_json"] = str(summary_path)
    return summary
