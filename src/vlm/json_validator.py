"""Robust parsing and validation for VLM JSON output."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.vlm.vlm_schema import VLMReview


def parse_vlm_review(text: str) -> tuple[VLMReview | None, dict[str, Any] | None]:
    """Parse VLM text into VLMReview or return a structured error dict."""
    try:
        if not text or not text.strip():
            raise ValueError("empty response")
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            cleaned = match.group(0)
        data = json.loads(cleaned)
        normalized = _normalize_review_data(data)
        return VLMReview.model_validate(normalized), None
    except (ValueError, json.JSONDecodeError, ValidationError) as exc:
        return None, {"error": "invalid_vlm_json", "message": str(exc), "raw": text[:500]}


def _normalize_review_data(data: Any) -> dict[str, Any]:
    """Normalize common VLM JSON drift before strict schema validation."""
    if not isinstance(data, dict):
        raise ValueError("VLM response JSON must be an object")

    normalized = dict(data)
    changed = False

    required_defaults: dict[str, Any] = {
        "is_anomaly": False,
        "event_type": "none",
        "alarm_level_suggestion": "none",
        "confidence": 0.0,
        "evidence_time": [],
        "evidence_tracks": [],
        "matched_rules": [],
        "reason": "No reason provided by VLM.",
        "possible_false_alarm": True,
        "recommended_action": "manual review recommended",
    }
    missing = [key for key in required_defaults if key not in normalized]
    if missing:
        raise ValueError(f"VLM response missing required fields: {missing}")

    for key, default in required_defaults.items():
        if normalized[key] is None:
            normalized[key] = default
            changed = True

    level = str(normalized.get("alarm_level_suggestion", "none")).lower()
    if level not in {"none", "low", "medium", "high"}:
        level = "none"
        changed = True
    normalized["alarm_level_suggestion"] = level

    try:
        confidence = float(normalized.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
        changed = True
    normalized["confidence"] = max(0.0, min(1.0, confidence))

    for key in ("evidence_time", "evidence_tracks", "matched_rules"):
        if not isinstance(normalized.get(key), list):
            normalized[key] = []
            changed = True

    normalized["is_anomaly"] = bool(normalized.get("is_anomaly", False))
    normalized["possible_false_alarm"] = bool(normalized.get("possible_false_alarm", False))
    normalized["event_type"] = str(normalized.get("event_type", "none") or "none")
    normalized["reason"] = str(normalized.get("reason", "No reason provided by VLM."))
    normalized["recommended_action"] = str(normalized.get("recommended_action", "manual review recommended"))
    metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
    if changed:
        metadata = {**metadata, "normalized": True}
    normalized["metadata"] = metadata
    return normalized
