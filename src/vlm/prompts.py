"""Prompt templates for structured VLM review."""

from __future__ import annotations

from typing import Any

from src.evidence.evidence_schema import EventEvidence, ObjectTrack


SYSTEM_PROMPT = """You are a surveillance anomaly review model.
Only judge from the provided structured evidence and keyframe paths.
Do not infer information outside the evidence.
If an image is provided, inspect the image directly, especially when detector results are empty.
Return JSON only. If evidence is insufficient, lower confidence.
Keep the JSON short. Do not include long explanations.
"""

OUTPUT_SCHEMA = """{
  "is_anomaly": true,
  "event_type": "restricted_area_intrusion",
  "alarm_level_suggestion": "high",
  "confidence": 0.87,
  "evidence_time": [[6.2, 14.8]],
  "evidence_tracks": [7],
  "matched_rules": ["person_entered_restricted_area", "dwell_time_over_8s"],
  "reason": "A person entered the restricted area and stayed near equipment for more than 8 seconds.",
  "possible_false_alarm": false,
  "recommended_action": "notify security staff and save the event clip"
}"""


def _trajectory_points(track: ObjectTrack) -> list[dict[str, Any]]:
    """Keep only start/middle/end trajectory points for prompt compactness."""
    if not track.trajectory:
        return []
    indexes = sorted({0, len(track.trajectory) // 2, len(track.trajectory) - 1})
    return [track.trajectory[index].model_dump(mode="json") for index in indexes]


def summarize_evidence(evidence: EventEvidence) -> dict[str, Any]:
    """Summarize evidence to reduce Qwen timeout probability."""
    tracks = []
    for track in evidence.objects[:10]:
        tracks.append(
            {
                "track_id": track.track_id,
                "label": track.label,
                "confidence": round(track.confidence, 4),
                "entered_rois": track.entered_rois,
                "dwell_time": round(track.dwell_time, 3),
                "direction": track.direction,
                "trajectory": _trajectory_points(track),
                "bbox_count": len(track.bboxes),
            }
        )
    return {
        "event_id": evidence.event_id,
        "camera_id": evidence.camera_id,
        "video_path": evidence.video_path,
        "time_range": evidence.time_range,
        "fps": evidence.fps,
        "roi_rules": [rule.model_dump(mode="json") for rule in evidence.roi_rules],
        "objects": tracks,
        "keyframes": [frame.model_dump(mode="json") for frame in evidence.keyframes[:8]],
        "windows": [window.model_dump(mode="json") for window in evidence.windows[:12]],
        "metadata": evidence.metadata,
    }


def build_review_prompt(evidence: EventEvidence) -> str:
    """Build a strict JSON prompt from structured event evidence."""
    compact = summarize_evidence(evidence)
    return (
        f"{SYSTEM_PROMPT}\n"
        "Review this structured temporal evidence and decide whether it is a real anomaly.\n"
        "Track Stream: object tracks, trajectories, ROI states, dwell time, and direction.\n"
        "SAMTracking Stream: if metadata.sam_tracking.summary exists, review railway track mask availability, "
        "object detections, mask-IoU intrusion score, suspicious frames, alarm frames, and sliding-window confirmation.\n"
        "Keyframe Stream: keyframe paths and boxes.\n"
        "Window Stream: short temporal window summaries.\n"
        "Return strictly one JSON object matching this schema:\n"
        f"{OUTPUT_SCHEMA}\n\n"
        f"event_evidence_summary:\n{compact}"
    )
