"""Tracker interfaces for STEAD perception modules."""

from __future__ import annotations

from typing import Protocol

from src.perception.detector import Detection


class Tracker(Protocol):
    """Object tracker protocol."""

    def update(self, detections: list[Detection]) -> dict[int, list[Detection]]:
        """Update tracks from detections and return active detection history."""

