"""Build short temporal window descriptions from structured evidence."""

from __future__ import annotations

from collections import Counter

from src.evidence.evidence_schema import EventEvidence, WindowDescription


class WindowBuilder:
    """Create fixed-size sliding window summaries."""

    def __init__(self, window_size: float = 4.0, stride: float = 2.0) -> None:
        self.window_size = window_size
        self.stride = stride

    def build(self, evidence: EventEvidence) -> list[WindowDescription]:
        """Build window summaries using tracks and triggered rules."""
        start, end = evidence.time_range
        windows: list[WindowDescription] = []
        current = start
        idx = 0
        while current < end or (idx == 0 and end == start):
            win_end = min(end, current + self.window_size)
            active_tracks: list[int] = []
            labels: Counter[str] = Counter()
            for track in evidence.objects:
                if any(current <= point.timestamp <= win_end for point in track.trajectory):
                    active_tracks.append(track.track_id)
                    labels[track.label] += 1
            triggered_rules = [
                rule.rule_id
                for rule in evidence.roi_rules
                if rule.triggered and rule.trigger_time and current <= rule.trigger_time[0] <= win_end
            ]
            summary = f"Window {current:.1f}-{win_end:.1f}s: {dict(labels)} active, rules={triggered_rules}."
            windows.append(
                WindowDescription(
                    window_id=f"w{idx:03d}",
                    start=current,
                    end=win_end,
                    visual_summary=summary,
                    object_count=dict(labels),
                    active_tracks=active_tracks,
                    triggered_rules=triggered_rules,
                )
            )
            idx += 1
            current += self.stride
            if end == start:
                break
        return windows

