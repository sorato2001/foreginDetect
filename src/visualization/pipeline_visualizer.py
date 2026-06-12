"""Visualization for detector, tracker, and ROI/rule outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.evidence.evidence_schema import EventEvidence, ObjectTrack


def save_pipeline_visualization(
    evidence: EventEvidence,
    rules_path: str,
    output_dir: str,
    video_path: str | None = None,
    image_path: str | None = None,
    max_video_frames: int | None = 300,
) -> dict[str, Any]:
    """Save visual artifacts for detector, tracker, and rule-engine stages."""
    output = Path(output_dir)
    vis_dir = output / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    overview_path = vis_dir / "pipeline_overview.jpg"
    video_output_path = vis_dir / "annotated_pipeline.mp4"
    animation_output_path = vis_dir / "evidence_animation.mp4"
    summary_path = vis_dir / "visualization_summary.json"

    summary = {
        "detector_stage": _detector_summary(evidence),
        "tracker_stage": _tracker_summary(evidence),
        "rule_stage": [rule.model_dump(mode="json") for rule in evidence.roi_rules],
        "artifacts": {
            "pipeline_overview": str(overview_path),
            "annotated_video": None,
            "evidence_animation": None,
        },
    }

    try:
        import cv2
        import numpy as np

        rules_config = _load_rules(rules_path)
        image = _load_background(video_path, evidence, cv2, np, image_path=image_path)
        _draw_rois(image, rules_config, cv2, np)
        _draw_tracks_and_boxes(image, evidence, cv2, timestamp=None)
        _draw_rule_status(image, evidence, cv2)
        cv2.imwrite(str(overview_path), image)
        annotated = _write_annotated_video(video_path, evidence, rules_config, video_output_path, cv2, np, max_video_frames=max_video_frames)
        if annotated:
            summary["artifacts"]["annotated_video"] = str(video_output_path)
        animation = _write_evidence_animation(evidence, rules_config, animation_output_path, cv2, np)
        if animation:
            summary["artifacts"]["evidence_animation"] = str(animation_output_path)
    except Exception as exc:  # pragma: no cover - visualization should never block pipeline
        summary["artifacts"]["pipeline_overview"] = None
        summary["error"] = f"{type(exc).__name__}: {exc}"

    _save_json(summary, summary_path)
    return {
        "overview_image": summary["artifacts"]["pipeline_overview"],
        "annotated_video": summary["artifacts"]["annotated_video"],
        "evidence_animation": summary["artifacts"]["evidence_animation"],
        "summary_json": str(summary_path),
    }


def save_image_visualization(
    evidence: EventEvidence,
    rules_path: str,
    output_dir: str,
    image_path: str,
) -> dict[str, Any]:
    """Save image-only visualization with rules and detections drawn on image."""
    output = Path(output_dir)
    vis_dir = output / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = vis_dir / "annotated_image.jpg"
    summary_path = vis_dir / "visualization_summary.json"

    summary = {
        "input_type": "image",
        "detector_stage": _detector_summary(evidence),
        "tracker_stage": _tracker_summary(evidence),
        "rule_stage": [rule.model_dump(mode="json") for rule in evidence.roi_rules],
        "artifacts": {"annotated_image": str(annotated_path)},
    }

    try:
        import cv2
        import numpy as np

        rules_config = _load_rules(rules_path)
        image = _load_background(None, evidence, cv2, np, image_path=image_path)
        _draw_rois(image, rules_config, cv2, np)
        _draw_tracks_and_boxes(image, evidence, cv2, timestamp=None)
        _draw_rule_status(image, evidence, cv2)
        cv2.imwrite(str(annotated_path), image)
    except Exception as exc:  # pragma: no cover - visualization should never block pipeline
        summary["artifacts"]["annotated_image"] = None
        summary["error"] = f"{type(exc).__name__}: {exc}"

    _save_json(summary, summary_path)
    return {"annotated_image": summary["artifacts"]["annotated_image"], "summary_json": str(summary_path)}


def _detector_summary(evidence: EventEvidence) -> dict[str, Any]:
    detections = []
    for track in evidence.objects:
        for bbox in track.bboxes:
            detections.append(
                {
                    "track_id": track.track_id,
                    "label": track.label,
                    "confidence": bbox.confidence,
                    "timestamp": bbox.timestamp,
                    "frame_index": bbox.frame_index,
                    "bbox": bbox.bbox,
                }
            )
    return {"total_detections": len(detections), "detections": detections[:100]}


def _tracker_summary(evidence: EventEvidence) -> dict[str, Any]:
    return {
        "total_tracks": len(evidence.objects),
        "tracks": [
            {
                "track_id": track.track_id,
                "label": track.label,
                "confidence": track.confidence,
                "trajectory_points": len(track.trajectory),
                "dwell_time": track.dwell_time,
                "start_time": track.trajectory[0].timestamp if track.trajectory else None,
                "end_time": track.trajectory[-1].timestamp if track.trajectory else None,
            }
            for track in evidence.objects
        ],
    }


def _load_background(video_path: str | None, evidence: EventEvidence, cv2: Any, np: Any, image_path: str | None = None) -> Any:
    if image_path and Path(image_path).exists():
        image = cv2.imread(image_path)
        if image is not None:
            return image
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return frame
    width, height = _canvas_size(evidence)
    return np.full((height, width, 3), 245, dtype=np.uint8)


def _canvas_size(evidence: EventEvidence) -> tuple[int, int]:
    max_x = 640.0
    max_y = 480.0
    for track in evidence.objects:
        for bbox in track.bboxes:
            max_x = max(max_x, bbox.bbox[2] + 80)
            max_y = max(max_y, bbox.bbox[3] + 80)
        for point in track.trajectory:
            max_x = max(max_x, point.x + 80)
            max_y = max(max_y, point.y + 80)
    return int(max_x), int(max_y)


def _load_rules(rules_path: str) -> dict[str, Any]:
    with open(rules_path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def _draw_rois(image: Any, rules_config: dict[str, Any], cv2: Any, np: Any) -> None:
    for roi in rules_config.get("rois", []):
        points = roi.get("points") or []
        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(image, [pts], True, (255, 120, 0), 2)
            cv2.putText(image, str(roi.get("roi_id", "roi")), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2)
    for rule in rules_config.get("rules", []):
        line = rule.get("line") or []
        if len(line) >= 2:
            pts = np.array(line, dtype=np.int32)
            cv2.line(image, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 2)
            cv2.putText(image, str(rule.get("rule_id", "line")), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def _draw_tracks_and_boxes(image: Any, evidence: EventEvidence, cv2: Any, timestamp: float | None = None) -> None:
    colors = [(0, 160, 255), (0, 200, 0), (220, 60, 60), (160, 60, 220), (50, 180, 180)]
    for track in evidence.objects:
        color = colors[(track.track_id - 1) % len(colors)]
        _draw_track(image, track, color, cv2, timestamp=timestamp)


def _draw_track(image: Any, track: ObjectTrack, color: tuple[int, int, int], cv2: Any, timestamp: float | None = None) -> None:
    visible_boxes = _visible_bboxes(track, timestamp)
    for bbox in visible_boxes:
        x1, y1, x2, y2 = [int(v) for v in bbox.bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"det {track.label} {bbox.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    points = [
        (int(point.x), int(point.y))
        for point in track.trajectory
        if timestamp is None or point.timestamp <= timestamp + 1e-6
    ]
    for idx, point in enumerate(points):
        cv2.circle(image, point, 4, color, -1)
        cv2.putText(image, f"T{track.track_id}", (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if idx > 0:
            cv2.line(image, points[idx - 1], point, color, 2)


def _visible_bboxes(track: ObjectTrack, timestamp: float | None) -> list[Any]:
    if timestamp is None:
        return track.bboxes
    if not track.bboxes:
        return []
    exact = [bbox for bbox in track.bboxes if abs(bbox.timestamp - timestamp) <= 0.35]
    if exact:
        return exact
    previous = [bbox for bbox in track.bboxes if bbox.timestamp <= timestamp]
    return previous[-1:] if previous else []


def _draw_rule_status(image: Any, evidence: EventEvidence, cv2: Any) -> None:
    y = 24
    cv2.putText(image, "STEAD: detector -> tracker -> ROI/rule engine", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    y += 28
    cv2.putText(image, f"tracks={len(evidence.objects)} rules={len(evidence.roi_rules)}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    y += 26
    for rule in evidence.roi_rules[:8]:
        color = (0, 0, 220) if rule.triggered else (80, 80, 80)
        status = "TRIGGERED" if rule.triggered else "off"
        text = f"{rule.rule_id}: {status} severity={rule.severity_hint} tracks={rule.evidence_tracks}"
        cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y += 22


def _write_annotated_video(
    video_path: str | None,
    evidence: EventEvidence,
    rules_config: dict[str, Any],
    output_path: Path,
    cv2: Any,
    np: Any,
    max_video_frames: int | None = 300,
) -> bool:
    """Write a frame-by-frame annotated video when an input video exists."""
    if not video_path or not Path(video_path).exists():
        return False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    fps = float(cap.get(cv2.CAP_PROP_FPS) or evidence.fps or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return False
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (width, height))
    frame_index = 0
    while True:
        if max_video_frames and frame_index >= max_video_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_index / fps if fps > 0 else 0.0
        _draw_rois(frame, rules_config, cv2, np)
        _draw_tracks_and_boxes(frame, evidence, cv2, timestamp=timestamp)
        _draw_rule_status(frame, evidence, cv2)
        cv2.putText(frame, f"t={timestamp:.2f}s frame={frame_index}", (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        writer.write(frame)
        frame_index += 1
    writer.release()
    cap.release()
    return output_path.exists() and output_path.stat().st_size > 0


def _write_evidence_animation(evidence: EventEvidence, rules_config: dict[str, Any], output_path: Path, cv2: Any, np: Any) -> bool:
    """Write a short synthetic animation from structured evidence."""
    width, height = _canvas_size(evidence)
    fps = 6.0
    duration = max(float(evidence.time_range[1] - evidence.time_range[0]), 1.0)
    frame_count = max(6, int(duration * fps))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame_idx in range(frame_count):
        timestamp = evidence.time_range[0] + (duration * frame_idx / max(1, frame_count - 1))
        frame = np.full((height, width, 3), 245, dtype=np.uint8)
        _draw_rois(frame, rules_config, cv2, np)
        _draw_tracks_and_boxes(frame, evidence, cv2, timestamp=timestamp)
        _draw_rule_status(frame, evidence, cv2)
        cv2.putText(frame, f"evidence animation t={timestamp:.2f}s", (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        writer.write(frame)
    writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def _save_json(data: dict[str, Any], path: Path) -> None:
    import json

    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
