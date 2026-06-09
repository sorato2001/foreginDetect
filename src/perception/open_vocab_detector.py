"""Open-vocabulary detector adapter placeholder."""

from __future__ import annotations

from typing import Any

from src.perception.detector import Detection


class OpenVocabularyDetector:
    """Adapter shell for Grounding DINO/YOLO-World style detectors."""

    def __init__(self, labels: list[str] | None = None) -> None:
        self.labels = labels or []

    def detect_frame(self, frame: Any, frame_index: int, timestamp: float) -> list[Detection]:
        """Return no detections until an open-vocabulary backend is configured."""
        return []

