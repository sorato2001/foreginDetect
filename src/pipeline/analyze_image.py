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
from src.evidence.serializers import save_json, save_model
from src.evidence.window_builder import WindowBuilder
from src.perception.detector import Detection
from src.perception.yolo_detector import YoloDetector
from src.pipeline.analyze_event import _configure_pipeline_logging, _pipeline_fallback_review, logger
from src.rules.rule_engine import RuleEngine
from src.vlm.mock_provider import MockVLMProvider
from src.vlm.qwen_provider import QwenProvider
from src.visualization.pipeline_visualizer import save_image_visualization


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


def run_image_pipeline(
    image_path: str,
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
    log_level: str = "INFO",
) -> dict[str, Any]:
    """Run STEAD analysis on a single image and write standard artifacts."""
    event_id = event_id or f"image_{uuid.uuid4().hex[:8]}"
    out = ensure_output_dir(output_dir)
    log_path = _configure_pipeline_logging(out, log_level=log_level)
    pipeline_start = time.perf_counter()
    logger.info(
        "STEP 01 start: event_id=%s camera_id=%s image=%s output=%s vlm_provider=%s",
        event_id,
        camera_id,
        image_path,
        out,
        vlm_provider,
    )
    logger.info("STEP 01 config: rules=%s mock_detections=%s save_visualization=%s", rules_path, mock_detections, save_visualization)

    step_start = time.perf_counter()
    logger.info("STEP 02 detector/tracker: begin image detection")
    tracks = _detect_image(image_path, mock_detections=mock_detections)
    detection_count = sum(len(history) for history in tracks.values())
    logger.info(
        "STEP 02 detector/tracker: done tracks=%s detections=%s elapsed=%.3fs",
        len(tracks),
        detection_count,
        time.perf_counter() - step_start,
    )

    step_start = time.perf_counter()
    logger.info("STEP 03 evidence: build single-image evidence")
    evidence = empty_evidence(event_id, camera_id, image_path, duration=1.0, fps=None)
    evidence.objects = build_object_tracks(tracks)
    evidence.metadata.update({"input_type": "image", "mock_detections": mock_detections, "stead_version": "stead_v1"})
    logger.info("STEP 03 evidence: done objects=%s elapsed=%.3fs", len(evidence.objects), time.perf_counter() - step_start)

    step_start = time.perf_counter()
    logger.info("STEP 04 ROI/rules: load rules and evaluate")
    rule_engine = RuleEngine.from_yaml(rules_path)
    evidence.roi_rules = rule_engine.evaluate(evidence)
    triggered_rules = [rule.rule_id for rule in evidence.roi_rules if rule.triggered]
    logger.info(
        "STEP 04 ROI/rules: done rules=%s triggered=%s elapsed=%.3fs",
        len(evidence.roi_rules),
        triggered_rules,
        time.perf_counter() - step_start,
    )

    step_start = time.perf_counter()
    logger.info("STEP 05 windows: build image window")
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
    logger.info("STEP 07 VLM: begin provider=%s timeout=%.1fs max_retries=%s", vlm_provider, vlm_timeout, vlm_max_retries)
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
        logger.exception("STEP 07 VLM: provider raised unexpectedly, using image pipeline fallback")
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
        "input_type": "image",
        "output_dir": str(out),
        "event_evidence": str(out / "event_evidence.json"),
        "vlm_review": str(out / "vlm_review.json"),
        "alarm_result": str(out / "alarm_result.json"),
        "visualization": visualization_artifacts,
        "pipeline_log": log_path,
        "final_level": alarm.final_level,
        "is_alarm": alarm.is_alarm,
    }
