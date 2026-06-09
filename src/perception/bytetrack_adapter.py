"""ByteTrack adapter placeholder with explicit fallback behavior."""

from __future__ import annotations

from src.perception.detector import Detection
from src.perception.simple_iou_tracker import SimpleIOUTracker


class ByteTrackAdapter(SimpleIOUTracker):
    """Fallback-compatible ByteTrack adapter.

    The class currently inherits the simple IOU implementation. A real ByteTrack
    backend can replace ``update`` without changing the pipeline contract.
    """

    def update(self, detections: list[Detection]) -> dict[int, list[Detection]]:
        """Update tracks using fallback logic."""
        return super().update(detections)

