"""Segmentation adapter interface."""

from __future__ import annotations

from typing import Any, Protocol


class Segmenter(Protocol):
    """Segmentation protocol."""

    def segment(self, frame: Any, boxes: list[list[float]]) -> list[Any]:
        """Return masks for input boxes."""

