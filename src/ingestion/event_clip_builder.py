"""Compatibility placeholder for event clip generation."""

from __future__ import annotations


class EventClipBuilder:
    """Adapter shell for FFmpeg clip building."""

    def __init__(self, ffmpeg_path: str = "ffmpeg") -> None:
        self.ffmpeg_path = ffmpeg_path

