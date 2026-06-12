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
        "rule_source": _rule_source(evidence),
        "detector_stage": _detector_summary(evidence),
        "tracker_stage": None if _uses_sam_mask_rule(evidence) else _tracker_summary(evidence),
        "sam_tracking_stage": _sam_tracking_summary(evidence) if _uses_sam_mask_rule(evidence) else None,
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
        sam_tracking = _load_sam_tracking(evidence)
        image = _load_background(video_path, evidence, cv2, np, image_path=image_path)
        if not _uses_sam_mask_rule(evidence):
            _draw_rois(image, rules_config, cv2, np)
            _draw_tracks_and_boxes(image, evidence, cv2, timestamp=None)
        _draw_sam_tracking_overlay(image, _nearest_sam_frame(sam_tracking, 0), cv2, np)
        _draw_rule_status(image, evidence, cv2)
        cv2.imwrite(str(overview_path), image)
        annotated = _write_annotated_video(
            video_path,
            evidence,
            rules_config,
            video_output_path,
            cv2,
            np,
            max_video_frames=max_video_frames,
            sam_tracking=sam_tracking,
        )
        if annotated:
            summary["artifacts"]["annotated_video"] = str(video_output_path)
        animation = _write_evidence_animation(evidence, rules_config, animation_output_path, cv2, np, sam_tracking=sam_tracking)
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
        "rule_source": _rule_source(evidence),
        "detector_stage": _detector_summary(evidence),
        "tracker_stage": None if _uses_sam_mask_rule(evidence) else _tracker_summary(evidence),
        "sam_tracking_stage": _sam_tracking_summary(evidence) if _uses_sam_mask_rule(evidence) else None,
        "rule_stage": [rule.model_dump(mode="json") for rule in evidence.roi_rules],
        "artifacts": {"annotated_image": str(annotated_path)},
    }

    try:
        import cv2
        import numpy as np

        rules_config = _load_rules(rules_path)
        image = _load_background(None, evidence, cv2, np, image_path=image_path)
        if not _uses_sam_mask_rule(evidence):
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


def _sam_tracking_summary(evidence: EventEvidence) -> dict[str, Any] | None:
    sam_meta = evidence.metadata.get("sam_tracking")
    if not isinstance(sam_meta, dict):
        return None
    return {
        "artifact": sam_meta.get("artifact"),
        "degraded": sam_meta.get("degraded"),
        "processed_frames": sam_meta.get("processed_frames"),
        "intrusion_events": sam_meta.get("intrusion_events"),
        "track_mask_seen": sam_meta.get("track_mask_seen"),
        "detections_seen": sam_meta.get("detections_seen"),
    }


def _load_sam_tracking(evidence: EventEvidence) -> dict[str, Any] | None:
    sam_meta = evidence.metadata.get("sam_tracking")
    if not isinstance(sam_meta, dict):
        return None
    artifact = sam_meta.get("artifact")
    if not artifact or not Path(str(artifact)).exists():
        return None
    try:
        import json

        with open(str(artifact), "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except Exception:
        return None


def _uses_sam_mask_rule(evidence: EventEvidence) -> bool:
    return evidence.metadata.get("rule_source") == "sam_tracking_mask_iou"


def _rule_source(evidence: EventEvidence) -> str:
    return str(evidence.metadata.get("rule_source", "config_rules"))


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


def _canvas_size(evidence: EventEvidence, sam_tracking: dict[str, Any] | None = None) -> tuple[int, int]:
    max_x = 640.0
    max_y = 480.0
    for track in evidence.objects:
        for bbox in track.bboxes:
            max_x = max(max_x, bbox.bbox[2] + 80)
            max_y = max(max_y, bbox.bbox[3] + 80)
        for point in track.trajectory:
            max_x = max(max_x, point.x + 80)
            max_y = max(max_y, point.y + 80)
    if sam_tracking:
        for frame in (sam_tracking.get("frames") or [])[:200]:
            for det in frame.get("detections") or []:
                bbox = det.get("bbox") or []
                if len(bbox) == 4:
                    max_x = max(max_x, float(bbox[2]) + 80)
                    max_y = max(max_y, float(bbox[3]) + 80)
            for contour in frame.get("track_mask_contours") or []:
                for x, y in contour:
                    max_x = max(max_x, float(x) + 80)
                    max_y = max(max_y, float(y) + 80)
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
    rule_source = _rule_source(evidence)
    cv2.putText(image, f"STEAD: detector -> tracker -> ROI/rule engine ({rule_source})", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
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
    sam_tracking: dict[str, Any] | None = None,
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
        if not _uses_sam_mask_rule(evidence):
            _draw_rois(frame, rules_config, cv2, np)
            _draw_tracks_and_boxes(frame, evidence, cv2, timestamp=timestamp)
        _draw_sam_tracking_overlay(frame, _nearest_sam_frame(sam_tracking, frame_index), cv2, np)
        _draw_rule_status(frame, evidence, cv2)
        cv2.putText(frame, f"t={timestamp:.2f}s frame={frame_index}", (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        writer.write(frame)
        frame_index += 1
    writer.release()
    cap.release()
    return output_path.exists() and output_path.stat().st_size > 0


def _nearest_sam_frame(sam_tracking: dict[str, Any] | None, frame_index: int) -> dict[str, Any] | None:
    if not sam_tracking:
        return None
    frames = sam_tracking.get("frames") or []
    if not frames:
        return None
    best = None
    best_gap = 10**9
    for item in frames:
        idx = int(item.get("frame_index", 0))
        gap = abs(idx - frame_index)
        if gap < best_gap:
            best = item
            best_gap = gap
        if idx > frame_index and gap > best_gap:
            break
    return best


def _draw_sam_tracking_overlay(image: Any, sam_frame: dict[str, Any] | None, cv2: Any, np: Any) -> None:
    """Overlay SAMTracking masks, detections, and intrusion status."""
    if not sam_frame:
        return
    overlay = image.copy()
    track_contours = sam_frame.get("track_mask_contours") or []
    for contour in track_contours:
        pts = np.array(contour, dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], (90, 80, 20))
            cv2.polylines(image, [pts], True, (120, 90, 20), 3)
    object_contours = sam_frame.get("object_mask_contours") or []
    for contour_group in object_contours:
        for contour in contour_group:
            pts = np.array(contour, dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], (0, 0, 220))
                cv2.polylines(image, [pts], True, (0, 0, 255), 2)
    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)

    for det in sam_frame.get("detections") or []:
        bbox = det.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (0, 0, 255) if sam_frame.get("alarm") else (0, 180, 255) if sam_frame.get("suspicious") else (0, 200, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label = f"{det.get('label', 'obj')} {float(det.get('confidence', 0.0)):.2f}"
        cv2.putText(image, label, (x1, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    h, w = image.shape[:2]
    alarm = bool(sam_frame.get("alarm"))
    suspicious = bool(sam_frame.get("suspicious"))
    max_iou = float(sam_frame.get("max_iou", 0.0))
    max_object_overlap = float(sam_frame.get("max_object_overlap", 0.0))
    window_count = int(sam_frame.get("window_count", 0))
    status = "ALARM" if alarm else "SUSPICIOUS" if suspicious else "NORMAL"
    color = (0, 0, 255) if alarm else (0, 180, 255) if suspicious else (0, 180, 0)
    if alarm:
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), color, 8)
    cv2.rectangle(image, (12, h - 76), (min(w - 12, 720), h - 14), (20, 20, 20), -1)
    cv2.putText(
        image,
        f"SAMTracking {status} | MaskIoU={max_iou:.3f} | ObjOverlap={max_object_overlap:.3f} | Window={window_count}",
        (24, h - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )
    if track_contours:
        cv2.putText(image, "railway track mask", (24, h - 94), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 90, 20), 2)


def _write_evidence_animation(
    evidence: EventEvidence,
    rules_config: dict[str, Any],
    output_path: Path,
    cv2: Any,
    np: Any,
    sam_tracking: dict[str, Any] | None = None,
) -> bool:
    """Write a short synthetic animation from structured evidence."""
    width, height = _canvas_size(evidence, sam_tracking=sam_tracking)
    fps = 6.0
    duration = max(float(evidence.time_range[1] - evidence.time_range[0]), 1.0)
    frame_count = max(6, int(duration * fps))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame_idx in range(frame_count):
        timestamp = evidence.time_range[0] + (duration * frame_idx / max(1, frame_count - 1))
        frame = np.full((height, width, 3), 245, dtype=np.uint8)
        if _uses_sam_mask_rule(evidence):
            source_frame_index = _frame_index_for_timestamp(sam_tracking, timestamp)
            _draw_sam_tracking_overlay(frame, _nearest_sam_frame(sam_tracking, source_frame_index), cv2, np)
        else:
            _draw_rois(frame, rules_config, cv2, np)
            _draw_tracks_and_boxes(frame, evidence, cv2, timestamp=timestamp)
        _draw_rule_status(frame, evidence, cv2)
        cv2.putText(frame, f"evidence animation t={timestamp:.2f}s", (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        writer.write(frame)
    writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def _frame_index_for_timestamp(sam_tracking: dict[str, Any] | None, timestamp: float) -> int:
    if not sam_tracking:
        return 0
    fps = float(sam_tracking.get("fps") or 25.0)
    return int(round(timestamp * fps))


def _save_json(data: dict[str, Any], path: Path) -> None:
    import json

    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
