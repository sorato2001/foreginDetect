"""Keyframe sampling for STEAD evidence generation."""

from __future__ import annotations

from pathlib import Path

from src.evidence.evidence_schema import EventEvidence, KeyframeInfo


class KeyframeSampler:
    """Sample representative keyframes from a video event."""

    def __init__(self, max_frames: int = 5) -> None:
        self.max_frames = max(1, max_frames)

    def sample(self, video_path: str, evidence: EventEvidence, output_dir: str) -> list[KeyframeInfo]:
        """Save first/middle/end frames and annotate selection reasons."""
        try:
            import cv2
        except Exception:
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        fps = float(cap.get(cv2.CAP_PROP_FPS) or evidence.fps or 25.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        positions = sorted({0, max(0, total // 2), max(0, total - 1), max(0, total // 4), max(0, (total * 3) // 4)})[: self.max_frames]
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        keyframes: list[KeyframeInfo] = []
        for idx, pos in enumerate(positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                continue
            timestamp = pos / fps if fps > 0 else float(idx)
            path = out_dir / f"keyframe_{idx:02d}_{pos}.jpg"
            if cv2.imwrite(str(path), frame):
                reason = "first_rule_trigger" if idx == 0 and any(r.triggered for r in evidence.roi_rules) else "uniform"
                keyframes.append(KeyframeInfo(timestamp=timestamp, frame_path=str(path), reason=reason, boxes=[]))
        cap.release()
        return keyframes

