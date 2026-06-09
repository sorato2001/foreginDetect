"""Compatibility placeholder for RTSP buffering.

The production RTSP/FFmpeg circular cache remains implemented in ``app.recorder``.
This module documents the future STEAD ingestion boundary without migrating the
legacy runtime.
"""

from __future__ import annotations


class RTSPBuffer:
    """Adapter shell for future RTSP cache integration."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id

