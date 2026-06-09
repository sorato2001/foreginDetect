"""Pydantic schemas for structured temporal event evidence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AlarmLevel = Literal["none", "low", "medium", "high"]


class FrameBBox(BaseModel):
    """Bounding box observed at a video timestamp."""

    timestamp: float
    frame_index: int | None = None
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TrajectoryPoint(BaseModel):
    """Track center point at a video timestamp."""

    timestamp: float
    x: float
    y: float
    roi_id: str | None = None


class ObjectTrack(BaseModel):
    """Object track summary used by rules and VLM prompts."""

    track_id: int
    label: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bboxes: list[FrameBBox] = Field(default_factory=list)
    trajectory: list[TrajectoryPoint] = Field(default_factory=list)
    entered_rois: list[str] = Field(default_factory=list)
    dwell_time: float = 0.0
    speed_stats: dict[str, float] = Field(default_factory=dict)
    direction: str | None = None


class ROIRuleTrigger(BaseModel):
    """Rule trigger result generated from ROI and track evidence."""

    rule_id: str
    rule_type: str
    roi_id: str | None = None
    triggered: bool = False
    trigger_time: list[float] | None = None
    evidence_tracks: list[int] = Field(default_factory=list)
    severity_hint: AlarmLevel = "none"


class KeyframeInfo(BaseModel):
    """Keyframe selected for visual review."""

    timestamp: float
    frame_path: str
    reason: str
    boxes: list[dict[str, Any]] = Field(default_factory=list)
    annotated_frame_path: str | None = None


class WindowDescription(BaseModel):
    """Short temporal window summary for prompt-level fusion."""

    window_id: str
    start: float
    end: float
    visual_summary: str | None = None
    object_count: dict[str, int] = Field(default_factory=dict)
    active_tracks: list[int] = Field(default_factory=list)
    triggered_rules: list[str] = Field(default_factory=list)


class EventEvidence(BaseModel):
    """Structured evidence JSON consumed by VLM and alarm fusion."""

    event_id: str
    camera_id: str
    video_path: str
    time_range: list[float] = Field(..., min_length=2, max_length=2)
    fps: float | None = None
    roi_rules: list[ROIRuleTrigger] = Field(default_factory=list)
    objects: list[ObjectTrack] = Field(default_factory=list)
    keyframes: list[KeyframeInfo] = Field(default_factory=list)
    windows: list[WindowDescription] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

