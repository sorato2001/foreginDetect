"""Prompt templates for structured VLM review."""

from __future__ import annotations

from src.evidence.evidence_schema import EventEvidence


SYSTEM_PROMPT = """You are a surveillance anomaly review model.
Only judge from the provided structured evidence and keyframe paths.
Do not infer information outside the evidence.
Return JSON only. If evidence is insufficient, lower confidence.
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


def build_review_prompt(evidence: EventEvidence) -> str:
    """Build a strict JSON prompt from structured event evidence."""
    compact = evidence.model_dump(mode="json")
    return (
        f"{SYSTEM_PROMPT}\n"
        "Review this structured temporal evidence and decide whether it is a real anomaly.\n"
        "Track Stream: object tracks, trajectories, ROI states, dwell time, and direction.\n"
        "Keyframe Stream: keyframe paths and boxes.\n"
        "Window Stream: short temporal window summaries.\n"
        "Return strictly one JSON object matching this schema:\n"
        f"{OUTPUT_SCHEMA}\n\n"
        f"event_evidence_json:\n{compact}"
    )

