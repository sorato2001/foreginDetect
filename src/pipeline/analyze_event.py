"""Command-line pipeline for STEAD event analysis."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

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
from src.vlm.pipeline_runner import run_vlm_review
from src.vlm.provider_factory import resolve_vlm_provider
from src.visualization.pipeline_visualizer import save_pipeline_visualization

logger = logging.getLogger("stead.pipeline")

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm"}
RUN_CONFIG_FILENAME = "run_config.json"
GUARD_NET_DEFAULT_PROMPT = "black chain link fence. black metal mesh fence. wire mesh fence. protective fence."


def _copy_batch_visualizations(result: dict[str, Any], batch_vis_dir: Path, prefix: str) -> list[dict[str, str]]:
    """Copy per-item visualization artifacts into the batch output directory."""
    visualization = result.get("visualization") if isinstance(result, dict) else None
    if not isinstance(visualization, dict):
        return []
    artifact_names = ["annotated_image", "annotated_video", "overview_image", "evidence_animation"]
    copied: list[dict[str, str]] = []
    batch_vis_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name in artifact_names:
        artifact_value = visualization.get(artifact_name)
        if not artifact_value:
            continue
        src = Path(str(artifact_value))
        if not src.exists() or not src.is_file():
            continue
        dst = batch_vis_dir / f"{prefix}_{artifact_name}{src.suffix}"
        shutil.copy2(src, dst)
        copied.append({"artifact": artifact_name, "source": str(src), "copy": str(dst)})
    return copied


def _parse_bool(value: str | bool) -> bool:
    """Parse CLI boolean values."""
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _parse_csv(value: str | None) -> list[str]:
    """Parse comma-separated CLI labels."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_output_dir(now: datetime | None = None) -> str:
    """Return a timestamped output directory that does not overwrite an existing run."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d%H%M%S")
    base = Path("outputs") / timestamp
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix:02d}")
        suffix += 1
    return str(candidate)


def _build_cli_run_config(args: argparse.Namespace, raw_argv: list[str]) -> dict[str, Any]:
    """Build a reproducible snapshot of parsed CLI arguments and effective settings."""
    arguments = dict(vars(args))
    effective_settings = dict(arguments)
    effective_settings.update(
        {
            "batch_limit": args.batch_limit or None,
            "max_analysis_frames": args.max_analysis_frames or None,
            "visualization_max_frames": args.visualization_max_frames or None,
            "vlm_fallback_on_error": _parse_bool(args.vlm_fallback_on_error),
            "save_visualization": not args.no_visualization,
            "sam_track_labels": _parse_csv(args.sam_track_labels),
            "sam_use_optical_flow": _parse_bool(args.sam_use_optical_flow),
            "sam2_enabled": _parse_bool(args.sam2_enabled),
        }
    )
    command_parts = ["python", "-m", "src.pipeline.analyze_event", *raw_argv]
    output_was_explicit = any(item == "--output" or item.startswith("--output=") for item in raw_argv)
    replay_parts = list(command_parts)
    if not output_was_explicit:
        replay_parts.extend(["--output", str(args.output)])
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": subprocess.list2cmdline(command_parts),
        "replay_command": subprocess.list2cmdline(replay_parts),
        "output_was_explicit": output_was_explicit,
        "arguments_include_defaults": True,
        "arguments": arguments,
        "effective_settings": effective_settings,
        "runtime": {
            "cwd": str(Path.cwd()),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def _save_cli_run_config(config: dict[str, Any], output_dirs: list[str | Path]) -> list[str]:
    """Write the CLI snapshot to each distinct output directory."""
    saved: list[str] = []
    seen: set[str] = set()
    for output_dir in output_dirs:
        path = Path(output_dir) / RUN_CONFIG_FILENAME
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        save_json(config, str(path))
        saved.append(str(path))
    return saved


def _result_output_dirs(result: dict[str, Any], root_output: str) -> list[str]:
    """Collect root and successful batch-item output directories."""
    output_dirs = [root_output]
    for item in result.get("results") or []:
        if isinstance(item, dict) and item.get("output_dir"):
            output_dirs.append(str(item["output_dir"]))
    return output_dirs


def _validate_rule_region_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fail fast for rule-region source combinations that cannot work."""
    source = args.rule_region_source
    if args.tracker != "sam_tracking" and source in {"sam_track", "guard_net"}:
        parser.error(f"rule-region-source={source} requires --tracker sam_tracking.")
    if source == "sam_track" and not args.sam_track_model:
        parser.error("rule-region-source=sam_track requires --sam-track-model.")
    if source == "yaml" and args.tracker == "simple_iou":
        return


def _guard_net_args_used(argv: list[str]) -> bool:
    """Return whether the user explicitly supplied any GuardNet option."""
    return any(item == "--rule-region-source=guard_net" or item.startswith("--guard-net-") for item in argv)


def _resolve_rule_region_source(
    requested: str,
    tracker: str,
    sam_track_model: str | None,
    guard_net_configured: bool = False,
) -> str:
    """Resolve auto into the first configured SAMTracking rule-region source."""
    requested = (requested or "auto").strip().lower()
    if tracker != "sam_tracking":
        return "yaml"
    if requested != "auto":
        return requested
    if guard_net_configured:
        return "guard_net"
    if sam_track_model:
        return "sam_track"
    return "yaml"


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


def collect_video_paths(video_path: str | Path, recursive: bool = False, limit: int | None = None) -> list[Path]:
    """Collect one video path or all supported videos in a directory."""
    root = Path(video_path)
    if root.is_file():
        return [root] if root.suffix.lower() in VIDEO_SUFFIXES else []
    if not root.is_dir():
        return []
    candidates = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(path for path in candidates if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    if limit is not None and limit > 0:
        return paths[:limit]
    return paths


def _batch_video_item_dir(output_dir: Path, source_root: Path, video_path: Path) -> Path:
    """Return a stable per-video output directory."""
    if source_root.is_dir():
        try:
            relative = video_path.relative_to(source_root).with_suffix("")
        except ValueError:
            relative = Path(video_path.stem)
    else:
        relative = Path(video_path.stem)
    safe_parts = [part.replace(" ", "_") for part in relative.parts]
    return output_dir.joinpath(*safe_parts)


def run_pipeline(
    video_path: str,
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
    max_analysis_frames: int | None = 900,
    visualization_max_frames: int | None = 300,
    log_level: str = "INFO",
    tracker: str = "simple_iou",
    sam_object_model: str | None = None,
    sam_track_model: str | None = None,
    sam_track_labels: list[str] | None = None,
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
    sam_temporal_mode: str = "fast",
    sam_sample_every: int = 15,
    sam_track_mask_interval: int = 30,
    sam_imgsz: int = 640,
    sam_progress_interval: int = 10,
    rule_region_source: str = "auto",
    guard_net_configured: bool = False,
    guard_net_backend: str = "grounding_dino",
    guard_net_text_prompt: str = GUARD_NET_DEFAULT_PROMPT,
    guard_net_model: str = "weights/groundingdino_swint_ogc.pth",
    guard_net_config: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    guard_net_checkpoint: str = "weights/groundingdino_swint_ogc.pth",
    guard_net_box_threshold: float = 0.10,
    guard_net_text_threshold: float = 0.10,
    guard_net_nms_threshold: float = 0.50,
    guard_net_max_box_area_ratio: float = 0.60,
    guard_net_scan_frames: int = 30,
    guard_net_sample_every: int = 5,
    guard_net_band_side_fraction: float = 0.08,
    guard_net_band_top_padding: int = 0,
    guard_net_band_bottom_padding: int = 0,
    guard_net_band_horizontal_padding: int = 0,
    guard_net_export_yolo_seg: bool = False,
    guard_net_yolo_class_id: int = 0,
) -> dict:
    """Run STEAD event analysis and write JSON artifacts."""
    event_id = event_id or f"event_{uuid.uuid4().hex[:8]}"
    out = ensure_output_dir(output_dir)
    log_path = _configure_pipeline_logging(out, log_level=log_level)
    pipeline_start = time.perf_counter()
    effective_vlm_provider = resolve_vlm_provider(vlm_provider, vlm_mode)
    logger.info(
        "STEP 01 start: event_id=%s camera_id=%s video=%s output=%s vlm_provider=%s vlm_mode=%s",
        event_id,
        camera_id,
        video_path,
        out,
        effective_vlm_provider,
        vlm_mode,
    )
    resolved_rule_region_source = _resolve_rule_region_source(rule_region_source, tracker, sam_track_model, guard_net_configured)
    logger.info(
        "STEP 01 config: rules=%s tracker=%s rule_region_source=%s resolved_rule_region_source=%s mock_detections=%s max_analysis_frames=%s save_visualization=%s visualization_max_frames=%s",
        rules_path,
        tracker,
        rule_region_source,
        resolved_rule_region_source,
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
            rule_region_source=resolved_rule_region_source,
            rule_region_source_requested=rule_region_source,
            track_mask_labels=sam_track_labels or [],
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
            temporal_mode=sam_temporal_mode,
            sample_every=sam_sample_every,
            track_mask_interval=sam_track_mask_interval,
            imgsz=sam_imgsz,
            progress_interval=sam_progress_interval,
            output_dir=str(out),
            guard_net_backend=guard_net_backend,
            guard_net_text_prompt=guard_net_text_prompt,
            guard_net_model=guard_net_model,
            guard_net_config=guard_net_config,
            guard_net_checkpoint=guard_net_checkpoint,
            guard_net_box_threshold=guard_net_box_threshold,
            guard_net_text_threshold=guard_net_text_threshold,
            guard_net_nms_threshold=guard_net_nms_threshold,
            guard_net_max_box_area_ratio=guard_net_max_box_area_ratio,
            guard_net_scan_frames=guard_net_scan_frames,
            guard_net_sample_every=guard_net_sample_every,
            guard_net_band_side_fraction=guard_net_band_side_fraction,
            guard_net_band_top_padding=guard_net_band_top_padding,
            guard_net_band_bottom_padding=guard_net_band_bottom_padding,
            guard_net_band_horizontal_padding=guard_net_band_horizontal_padding,
            guard_net_export_yolo_seg=guard_net_export_yolo_seg,
            guard_net_yolo_class_id=guard_net_yolo_class_id,
            sam_track_fallback_enabled=rule_region_source == "auto" and bool(sam_track_model),
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
    logger.info("STEP 05 windows: build temporal windows")
    evidence.windows = WindowBuilder().build(evidence)
    evidence.metadata.update({"mock_detections": mock_detections, "stead_version": "stead_v1", "tracker": tracker})
    sam_tracking_artifact: str | None = None
    if sam_tracking_result is not None:
        sam_tracking_artifact = str(out / "sam_tracking_result.json")
        save_json(sam_tracking_result.to_json_dict(), sam_tracking_artifact)
        evidence.metadata["sam_tracking"] = _sam_tracking_evidence_metadata(sam_tracking_result, sam_tracking_artifact)
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
        "STEP 07 VLM: begin provider=%s mode=%s timeout=%.1fs max_retries=%s fallback_on_error=%s",
        effective_vlm_provider,
        vlm_mode,
        vlm_timeout,
        vlm_max_retries,
        vlm_fallback_on_error,
    )
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
        local_evidence_mode="qwen",
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


def run_video_batch_pipeline(
    video_path: str,
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
    max_analysis_frames: int | None = 900,
    visualization_max_frames: int | None = 300,
    log_level: str = "INFO",
    tracker: str = "simple_iou",
    sam_object_model: str | None = None,
    sam_track_model: str | None = None,
    sam_track_labels: list[str] | None = None,
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
    sam_temporal_mode: str = "fast",
    sam_sample_every: int = 15,
    sam_track_mask_interval: int = 30,
    sam_imgsz: int = 640,
    sam_progress_interval: int = 10,
    rule_region_source: str = "auto",
    guard_net_configured: bool = False,
    guard_net_backend: str = "grounding_dino",
    guard_net_text_prompt: str = GUARD_NET_DEFAULT_PROMPT,
    guard_net_model: str = "weights/groundingdino_swint_ogc.pth",
    guard_net_config: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    guard_net_checkpoint: str = "weights/groundingdino_swint_ogc.pth",
    guard_net_box_threshold: float = 0.10,
    guard_net_text_threshold: float = 0.10,
    guard_net_nms_threshold: float = 0.50,
    guard_net_max_box_area_ratio: float = 0.60,
    guard_net_scan_frames: int = 30,
    guard_net_sample_every: int = 5,
    guard_net_band_side_fraction: float = 0.08,
    guard_net_band_top_padding: int = 0,
    guard_net_band_bottom_padding: int = 0,
    guard_net_band_horizontal_padding: int = 0,
    guard_net_export_yolo_seg: bool = False,
    guard_net_yolo_class_id: int = 0,
    recursive: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run video analysis for a file or directory and write a batch summary."""
    source = Path(video_path)
    out = ensure_output_dir(output_dir)
    video_paths = collect_video_paths(source, recursive=recursive, limit=limit)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    batch_visualizations: list[dict[str, str]] = []
    batch_vis_dir = out / "batch_visualizations"

    for index, item in enumerate(video_paths, start=1):
        item_output = _batch_video_item_dir(out, source, item)
        event_id = f"video_{item.stem}_{index:04d}"
        try:
            result = run_pipeline(
                video_path=str(item),
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
                max_analysis_frames=max_analysis_frames,
                visualization_max_frames=visualization_max_frames,
                log_level=log_level,
                tracker=tracker,
                sam_object_model=sam_object_model,
                sam_track_model=sam_track_model,
                sam_track_labels=sam_track_labels,
                sam2_config=sam2_config,
                sam2_checkpoint=sam2_checkpoint,
                sam_device=sam_device,
                sam_iou_threshold=sam_iou_threshold,
                sam_object_overlap_threshold=sam_object_overlap_threshold,
                sam_window_size=sam_window_size,
                sam_confirm_count=sam_confirm_count,
                sam_use_optical_flow=sam_use_optical_flow,
                sam2_enabled=sam2_enabled,
                sam2_scan_frames=sam2_scan_frames,
                sam2_prompt_mode=sam2_prompt_mode,
                sam_temporal_mode=sam_temporal_mode,
                sam_sample_every=sam_sample_every,
                sam_track_mask_interval=sam_track_mask_interval,
                sam_imgsz=sam_imgsz,
                sam_progress_interval=sam_progress_interval,
                rule_region_source=rule_region_source,
                guard_net_configured=guard_net_configured,
                guard_net_backend=guard_net_backend,
                guard_net_text_prompt=guard_net_text_prompt,
                guard_net_model=guard_net_model,
                guard_net_config=guard_net_config,
                guard_net_checkpoint=guard_net_checkpoint,
                guard_net_box_threshold=guard_net_box_threshold,
                guard_net_text_threshold=guard_net_text_threshold,
                guard_net_nms_threshold=guard_net_nms_threshold,
                guard_net_max_box_area_ratio=guard_net_max_box_area_ratio,
                guard_net_scan_frames=guard_net_scan_frames,
                guard_net_sample_every=guard_net_sample_every,
                guard_net_band_side_fraction=guard_net_band_side_fraction,
                guard_net_band_top_padding=guard_net_band_top_padding,
                guard_net_band_bottom_padding=guard_net_band_bottom_padding,
                guard_net_band_horizontal_padding=guard_net_band_horizontal_padding,
                guard_net_export_yolo_seg=guard_net_export_yolo_seg,
                guard_net_yolo_class_id=guard_net_yolo_class_id,
            )
            result["source_video"] = str(item)
            copied_visualizations = _copy_batch_visualizations(result, batch_vis_dir, f"{index:04d}_{item.stem}")
            result["batch_visualizations"] = copied_visualizations
            batch_visualizations.extend(copied_visualizations)
            results.append(result)
        except Exception as exc:
            logger.exception("Batch video analysis failed: video=%s", item)
            failures.append({"video": str(item), "error_type": type(exc).__name__, "error_message": str(exc)[:500]})

    summary = {
        "input_type": "video_batch",
        "source": str(source),
        "output_dir": str(out),
        "recursive": recursive,
        "total": len(video_paths),
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


def _sam_tracking_mask_iou_rule(sam_tracking_result: SAMTrackingResult | None) -> ROIRuleTrigger | None:
    """Build a STEP 04 rule result from SAMTracking mask-IoU intrusion judgment."""
    if sam_tracking_result is None or not _sam_tracking_has_track_mask(sam_tracking_result):
        return None
    source = sam_tracking_result.metadata.get("rule_region_source_resolved") or "sam_track"
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
    rule_id = "sam_track_mask_iou_intrusion"
    roi_id = "sam_track_rule_region"
    return ROIRuleTrigger(
        rule_id=rule_id,
        rule_type="mask_iou_intrusion",
        roi_id=roi_id,
        triggered=triggered,
        trigger_time=trigger_time,
        evidence_tracks=evidence_tracks,
        severity_hint=severity,
    )


def _sam_tracking_rule_source(sam_tracking_result: SAMTrackingResult | None) -> str:
    """Return a display-safe rule source for segmentation mask IoU rules."""
    if sam_tracking_result is None:
        return "sam_tracking_mask_iou"
    source = sam_tracking_result.metadata.get("rule_region_source_resolved")
    if source == "sam_track":
        return "sam_track_mask_iou"
    return "sam_tracking_mask_iou"


def _sam_tracking_has_track_mask(sam_tracking_result: SAMTrackingResult) -> bool:
    """Return true only when SAMTracking produced an actual railway/track mask."""
    if sam_tracking_result.metadata.get("rule_region_source_resolved") == "yaml":
        return False
    if not sam_tracking_result.metadata.get("track_mask_seen"):
        return False
    return any(frame.track_mask_available for frame in sam_tracking_result.frames)


def _sam_tracking_prompt_summary(sam_tracking_result: SAMTrackingResult) -> dict:
    """Build a compact SAMTracking summary for VLM review."""
    frames = sam_tracking_result.frames
    suspicious = [frame for frame in frames if frame.suspicious]
    alarms = [frame for frame in frames if frame.alarm]
    peak = max(frames, key=lambda frame: max(frame.max_iou, frame.max_object_overlap), default=None)
    sample_frames = alarms[:3] or suspicious[:3] or ([peak] if peak is not None else [])
    return {
        "rule_type": "mask_iou_intrusion",
        "rule_region_source": sam_tracking_result.metadata.get("rule_region_source_resolved"),
        "rule_region_type": "segmentation_mask",
        "intrusion_judgment": "mask_iou_and_object_overlap",
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


def _sam_tracking_evidence_metadata(sam_tracking_result: SAMTrackingResult, artifact: str) -> dict[str, Any]:
    """Expose rule-region geometry and intrusion facts to evidence/VLM consumers."""
    metadata = sam_tracking_result.metadata
    guard_net = metadata.get("guard_net") or {}
    geometry = guard_net.get("continuous_band") or {}
    summary = _sam_tracking_prompt_summary(sam_tracking_result)
    actual_rule_region_available = _sam_tracking_has_track_mask(sam_tracking_result)
    events = [
        {
            "event_id": item.event_id,
            "start_frame": item.start_frame,
            "end_frame": item.end_frame,
            "peak_iou": item.peak_iou,
            "frames_suspicious": item.frames_suspicious,
        }
        for item in sam_tracking_result.intrusion_events
    ]
    return {
        "artifact": artifact,
        "degraded": metadata.get("degraded"),
        "processed_frames": metadata.get("processed_frames"),
        "intrusion_events": len(events),
        "track_mask_seen": actual_rule_region_available,
        "rule_region_source": metadata.get("rule_region_source_resolved"),
        "rule_region_available": actual_rule_region_available,
        "detections_seen": metadata.get("detections_seen"),
        "rule_region": {
            "source": metadata.get("rule_region_source_resolved"),
            "available": actual_rule_region_available,
            "text_prompt": guard_net.get("text_prompt"),
            "source_frame_index": guard_net.get("source_frame_index", guard_net.get("source_frame_indices")),
            "confidence": guard_net.get("aggregate_confidence", guard_net.get("selected_candidate_score")),
            "continuous_band_polygon": geometry.get("polygon"),
            "continuous_band_polygons": geometry.get("polygons", []),
            "continuous_band_polygon_count": geometry.get("polygon_count", 0),
            "continuous_band_area": geometry.get("continuous_mask_area"),
        },
        "person_mask_intrusion_evidence": {
            "max_mask_iou": summary["max_iou"],
            "max_object_overlap": summary["max_object_overlap"],
            "suspicious_frame_count": summary["suspicious_frame_count"],
            "confirmed_alarm_frame_count": summary["alarm_frame_count"],
            "intrusion_events": events,
        },
        "summary": summary,
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze one surveillance event with STEAD.")
    parser.add_argument("--input-type", choices=["video", "image"], default="video")
    parser.add_argument("--video", default=None)
    parser.add_argument("--video-dir", default=None, help="Directory of videos for batch video analysis")
    parser.add_argument("--image", default=None)
    parser.add_argument("--recursive", action="store_true", help="For input directories, scan subdirectories")
    parser.add_argument("--batch-limit", type=int, default=0, help="For input directories, limit number of files; 0 means no limit")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--rules", default="configs/rules.example.yaml")
    parser.add_argument("--output", default=None, help="Output directory; defaults to outputs/YYYYMMDDHHMMSS")
    parser.add_argument("--vlm-provider", choices=["auto", "mock", "qwen", "gemma", "local_gemma"], default="auto")
    parser.add_argument("--vlm-mode", "--vlm_mode", dest="vlm_mode", choices=["local", "web"], default="web", help="local uses LAN Gemma VLM; web uses Qwen/DashScope")
    parser.add_argument("--vlm-local-endpoint", default="http://localhost:8082/v1/chat/completions")
    parser.add_argument("--vlm-local-model", default="gemma-4-26B")
    parser.add_argument("--vlm-local-max-images", type=int, default=1)
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
    parser.add_argument("--rule-region-source", choices=["yaml", "sam_track", "guard_net", "auto"], default="auto")
    parser.add_argument("--sam-object-model", default=None, help="YOLO11 object model path for SAMTracking")
    parser.add_argument("--sam-track-model", default=None, help="Rail/track segmentation best.pt path for SAMTracking")
    parser.add_argument("--sam-track-labels", default=None, help="Comma-separated segmentation labels to merge as railway/track mask")
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
    parser.add_argument("--sam-temporal-mode", choices=["fast", "faithful", "reference"], default="fast", help="SAMTracking video timing: faithful/reference runs YOLO and railway mask every frame; fast caches them while SAM2 still advances every frame")
    parser.add_argument("--sam-sample-every", type=int, default=15, help="Run object detection every N frames in SAMTracking")
    parser.add_argument("--sam-track-mask-interval", type=int, default=30, help="Run railway mask segmentation every N frames in SAMTracking")
    parser.add_argument("--sam-imgsz", type=int, default=640, help="YOLO inference image size for SAMTracking")
    parser.add_argument("--sam-progress-interval", type=int, default=10, help="Log SAMTracking progress every N processed frames")
    parser.add_argument("--guard-net-backend", choices=["grounding_dino", "yoloe"], default="grounding_dino")
    parser.add_argument("--guard-net-text-prompt", default=GUARD_NET_DEFAULT_PROMPT)
    parser.add_argument("--guard-net-model", default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--guard-net-config", default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--guard-net-checkpoint", default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--guard-net-box-threshold", type=float, default=0.10)
    parser.add_argument("--guard-net-text-threshold", type=float, default=0.10)
    parser.add_argument("--guard-net-nms-threshold", type=float, default=0.50)
    parser.add_argument("--guard-net-max-box-area-ratio", type=float, default=0.60)
    parser.add_argument("--guard-net-scan-frames", type=int, default=30)
    parser.add_argument("--guard-net-sample-every", type=int, default=5)
    parser.add_argument("--guard-net-band-side-fraction", type=float, default=0.08)
    parser.add_argument("--guard-net-band-top-padding", type=int, default=0)
    parser.add_argument("--guard-net-band-bottom-padding", type=int, default=0)
    parser.add_argument("--guard-net-band-horizontal-padding", type=int, default=0)
    parser.add_argument("--guard-net-export-yolo-seg", default="false")
    parser.add_argument("--guard-net-yolo-class-id", type=int, default=0)
    args = parser.parse_args()
    raw_argv = sys.argv[1:]
    guard_net_configured = _guard_net_args_used(raw_argv)
    guard_net_kwargs = {
        "guard_net_configured": guard_net_configured,
        "guard_net_backend": args.guard_net_backend,
        "guard_net_text_prompt": args.guard_net_text_prompt,
        "guard_net_model": args.guard_net_model,
        "guard_net_config": args.guard_net_config,
        "guard_net_checkpoint": args.guard_net_checkpoint,
        "guard_net_box_threshold": args.guard_net_box_threshold,
        "guard_net_text_threshold": args.guard_net_text_threshold,
        "guard_net_nms_threshold": args.guard_net_nms_threshold,
        "guard_net_max_box_area_ratio": args.guard_net_max_box_area_ratio,
        "guard_net_scan_frames": args.guard_net_scan_frames,
        "guard_net_sample_every": args.guard_net_sample_every,
        "guard_net_band_side_fraction": args.guard_net_band_side_fraction,
        "guard_net_band_top_padding": args.guard_net_band_top_padding,
        "guard_net_band_bottom_padding": args.guard_net_band_bottom_padding,
        "guard_net_band_horizontal_padding": args.guard_net_band_horizontal_padding,
        "guard_net_export_yolo_seg": _parse_bool(args.guard_net_export_yolo_seg),
        "guard_net_yolo_class_id": args.guard_net_yolo_class_id,
    }
    try:
        resolve_vlm_provider(args.vlm_provider, args.vlm_mode)
    except ValueError as exc:
        parser.error(str(exc))
    _validate_rule_region_args(parser, args)
    if args.input_type == "image" and not args.image:
        parser.error("--image is required when --input-type image")
    if args.input_type == "video" and not (args.video_dir or args.video):
        parser.error("--video or --video-dir is required when --input-type video")
    if args.output is None:
        args.output = _default_output_dir()
    run_config = _build_cli_run_config(args, raw_argv)
    _save_cli_run_config(run_config, [args.output])
    if args.input_type == "image":
        image_source = Path(args.image)
        if image_source.is_dir():
            from src.pipeline.analyze_image import run_image_batch_pipeline

            result = run_image_batch_pipeline(
                image_path=args.image,
                camera_id=args.camera_id,
                rules_path=args.rules,
                output_dir=args.output,
                vlm_provider=args.vlm_provider,
                vlm_mode=args.vlm_mode,
                vlm_local_endpoint=args.vlm_local_endpoint,
                vlm_local_model=args.vlm_local_model,
                vlm_local_max_images=args.vlm_local_max_images,
                mock_detections=args.mock_detections,
                vlm_timeout=args.vlm_timeout,
                vlm_max_retries=args.vlm_max_retries,
                vlm_fallback_on_error=_parse_bool(args.vlm_fallback_on_error),
                save_visualization=not args.no_visualization,
                log_level=args.log_level,
                tracker=args.tracker,
                sam_object_model=args.sam_object_model,
                sam_track_model=args.sam_track_model,
                sam_track_labels=_parse_csv(args.sam_track_labels),
                sam2_config=args.sam2_config,
                sam2_checkpoint=args.sam2_checkpoint,
                sam2_enabled=_parse_bool(args.sam2_enabled),
                sam_device=args.sam_device,
                sam_iou_threshold=args.sam_iou_threshold,
                sam_object_overlap_threshold=args.sam_object_overlap_threshold,
                sam_imgsz=args.sam_imgsz,
                rule_region_source=args.rule_region_source,
                recursive=args.recursive,
                limit=args.batch_limit or None,
                **guard_net_kwargs,
            )
        else:
            from src.pipeline.analyze_image import run_image_pipeline

            result = run_image_pipeline(
                image_path=args.image,
                camera_id=args.camera_id,
                rules_path=args.rules,
                output_dir=args.output,
                vlm_provider=args.vlm_provider,
                vlm_mode=args.vlm_mode,
                vlm_local_endpoint=args.vlm_local_endpoint,
                vlm_local_model=args.vlm_local_model,
                vlm_local_max_images=args.vlm_local_max_images,
                event_id=args.event_id,
                mock_detections=args.mock_detections,
                vlm_timeout=args.vlm_timeout,
                vlm_max_retries=args.vlm_max_retries,
                vlm_fallback_on_error=_parse_bool(args.vlm_fallback_on_error),
                save_visualization=not args.no_visualization,
                log_level=args.log_level,
                tracker=args.tracker,
                sam_object_model=args.sam_object_model,
                sam_track_model=args.sam_track_model,
                sam_track_labels=_parse_csv(args.sam_track_labels),
                sam2_config=args.sam2_config,
                sam2_checkpoint=args.sam2_checkpoint,
                sam2_enabled=_parse_bool(args.sam2_enabled),
                sam_device=args.sam_device,
                sam_iou_threshold=args.sam_iou_threshold,
                sam_object_overlap_threshold=args.sam_object_overlap_threshold,
                sam_imgsz=args.sam_imgsz,
                rule_region_source=args.rule_region_source,
                **guard_net_kwargs,
            )
    else:
        video_source_arg = args.video_dir or args.video
        video_source = Path(video_source_arg)
        if video_source.is_dir():
            result = run_video_batch_pipeline(
                video_path=video_source_arg,
                camera_id=args.camera_id,
                rules_path=args.rules,
                output_dir=args.output,
                vlm_provider=args.vlm_provider,
                vlm_mode=args.vlm_mode,
                vlm_local_endpoint=args.vlm_local_endpoint,
                vlm_local_model=args.vlm_local_model,
                vlm_local_max_images=args.vlm_local_max_images,
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
                sam_track_labels=_parse_csv(args.sam_track_labels),
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
                sam_temporal_mode=args.sam_temporal_mode,
                sam_sample_every=args.sam_sample_every,
                sam_track_mask_interval=args.sam_track_mask_interval,
                sam_imgsz=args.sam_imgsz,
                sam_progress_interval=args.sam_progress_interval,
                rule_region_source=args.rule_region_source,
                recursive=args.recursive,
                limit=args.batch_limit or None,
                **guard_net_kwargs,
            )
        else:
            result = run_pipeline(
                video_path=video_source_arg,
                camera_id=args.camera_id,
                rules_path=args.rules,
                output_dir=args.output,
                vlm_provider=args.vlm_provider,
                vlm_mode=args.vlm_mode,
                vlm_local_endpoint=args.vlm_local_endpoint,
                vlm_local_model=args.vlm_local_model,
                vlm_local_max_images=args.vlm_local_max_images,
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
                sam_track_labels=_parse_csv(args.sam_track_labels),
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
                sam_temporal_mode=args.sam_temporal_mode,
                sam_sample_every=args.sam_sample_every,
                sam_track_mask_interval=args.sam_track_mask_interval,
                sam_imgsz=args.sam_imgsz,
                sam_progress_interval=args.sam_progress_interval,
                rule_region_source=args.rule_region_source,
                **guard_net_kwargs,
            )
    run_config_paths = _save_cli_run_config(run_config, _result_output_dirs(result, args.output))
    result["run_config"] = run_config_paths[0]
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
