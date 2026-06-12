"""Event stream buffering adapter shell.

STEAD operates on event clips by default. This adapter marks the boundary where
a live RTSP or cached-video ingestion backend can be connected later.
"""

from __future__ import annotations


class RTSPBuffer:
    """Adapter shell for future video stream buffering."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
