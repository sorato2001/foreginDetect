"""Detector interfaces used by the STEAD pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class Detection:
    """Single frame detection."""

    label: str
    confidence: float
    bbox: list[float]
    frame_index: int = 0
    timestamp: float = 0.0


class Detector(Protocol):
    """Object detector protocol."""

    def detect_frame(self, frame, frame_index: int, timestamp: float) -> list[Detection]:
        """Detect objects from one frame."""

