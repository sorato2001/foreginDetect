"""Pydantic schema for final graded alarm output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AlarmLevel = Literal["none", "low", "medium", "high"]


class AlarmResult(BaseModel):
    """Final alarm fusion result for storage and API responses."""

    event_id: str
    final_level: AlarmLevel
    final_score: float = Field(ge=0.0, le=1.0)
    is_alarm: bool
    reasons: list[str]
    evidence_time: list[list[float]] = Field(default_factory=list)
    evidence_tracks: list[int] = Field(default_factory=list)
    rule_score: float = Field(ge=0.0, le=1.0)
    vlm_score: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    action: str

