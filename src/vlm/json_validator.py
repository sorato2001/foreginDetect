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
        return VLMReview.model_validate(data), None
    except (ValueError, json.JSONDecodeError, ValidationError) as exc:
        return None, {"error": "invalid_vlm_json", "message": str(exc), "raw": text[:500]}

