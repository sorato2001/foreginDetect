"""Utilities for building EventEvidence from tracked detections."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.evidence.evidence_schema import EventEvidence, FrameBBox, ObjectTrack, TrajectoryPoint
from src.perception.detector import Detection


def build_object_tracks(tracks: dict[int, list[Detection]]) -> list[ObjectTrack]:
    """Convert tracker histories into structured ObjectTrack objects."""
    objects: list[ObjectTrack] = []
    for track_id, history in sorted(tracks.items()):
        if not history:
            continue
        labels = Counter(det.label for det in history)
        label = labels.most_common(1)[0][0]
        confidence = sum(det.confidence for det in history) / len(history)
        bboxes = [
            FrameBBox(
                timestamp=det.timestamp,
                frame_index=det.frame_index,
                bbox=det.bbox,
                confidence=det.confidence,
            )
            for det in history
        ]
        trajectory = [
            TrajectoryPoint(
                timestamp=det.timestamp,
                x=(det.bbox[0] + det.bbox[2]) / 2.0,
                y=(det.bbox[1] + det.bbox[3]) / 2.0,
            )
            for det in history
        ]
        dwell_time = max(0.0, trajectory[-1].timestamp - trajectory[0].timestamp) if len(trajectory) >= 2 else 0.0
        objects.append(
            ObjectTrack(
                track_id=track_id,
                label=label,
                confidence=confidence,
                bboxes=bboxes,
                trajectory=trajectory,
                dwell_time=dwell_time,
                speed_stats={},
                direction=None,
            )
        )
    return objects


def empty_evidence(event_id: str, camera_id: str, video_path: str, duration: float = 0.0, fps: float | None = None) -> EventEvidence:
    """Create a valid empty evidence object."""
    return EventEvidence(
        event_id=event_id,
        camera_id=camera_id,
        video_path=str(Path(video_path)),
        time_range=[0.0, duration],
        fps=fps,
        metadata={"pipeline": "stead_v1", "fallback": True},
    )

