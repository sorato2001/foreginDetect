"""Pydantic schema for VLM review output."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AlarmLevel = Literal["none", "low", "medium", "high"]


class VLMReview(BaseModel):
    """Strict JSON-compatible result produced by a review provider."""

    is_anomaly: bool
    event_type: str
    alarm_level_suggestion: AlarmLevel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_time: list[list[float]] = Field(default_factory=list)
    evidence_tracks: list[int] = Field(default_factory=list)
    matched_rules: list[str] = Field(default_factory=list)
    reason: str
    possible_false_alarm: bool = False
    recommended_action: str
    metadata: dict[str, Any] = Field(default_factory=dict)
