"""SAM/SAM2 adapter placeholder."""

from __future__ import annotations

from typing import Any


class SamAdapter:
    """No-op SAM adapter that keeps bbox-only experiments runnable."""

    def segment(self, frame: Any, boxes: list[list[float]]) -> list[Any]:
        """Return an empty mask list when SAM is unavailable."""
        return []

