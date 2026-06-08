"""Frame sampling utilities for event-level railway security review.

The VLM reviewer should only see a few representative images, never the full
real-time stream. This module extracts at most five frames from an event video.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


class FrameSampler:
    """Sample representative frames from a video for offline review."""

    def __init__(self, max_frames: int = 5) -> None:
        self.max_frames = max(1, int(max_frames))

    def sample_video(
        self,
        video_path: str,
        output_dir: str,
        prefix: str = "keyframe",
        include_paths: Iterable[str] | None = None,
    ) -> list[str]:
        """Extract beginning/middle/end frames and return saved image paths."""
        frames: list[str] = []
        for path in include_paths or []:
            if path and Path(path).exists() and path not in frames:
                frames.append(path)
                if len(frames) >= self.max_frames:
                    return frames

        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning("Cannot open video for frame sampling: %s", video_path)
                return frames
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total <= 0:
                positions = [0]
            else:
                positions = sorted({0, max(0, total // 2), max(0, total - 1), max(0, total // 4), max(0, (total * 3) // 4)})
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for pos in positions:
                if len(frames) >= self.max_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                out_path = out_dir / f"{prefix}_{pos}.jpg"
                if cv2.imwrite(str(out_path), frame):
                    frames.append(str(out_path))
            cap.release()
        except Exception as exc:  # pragma: no cover - depends on cv2/video runtime
            logger.error("Failed to sample frames from %s: %s", video_path, exc, exc_info=True)
        return frames[: self.max_frames]
