"""Rule engine for structured temporal surveillance evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import yaml

from src.evidence.evidence_schema import EventEvidence, ObjectTrack, ROIRuleTrigger
from src.rules.roi import distance, point_in_polygon, signed_line_side


class RuleEngine:
    """Evaluate ROI, line, dwell, gathering, and proximity rules."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config or {}
        self.rois = {item["roi_id"]: item for item in self.config.get("rois", [])}
        self.rules = list(self.config.get("rules", []))

    @classmethod
    def from_yaml(cls, path: str) -> "RuleEngine":
        """Load rule configuration from YAML."""
        with open(path, "r", encoding="utf-8") as file_obj:
            data = yaml.safe_load(file_obj) or {}
        return cls(data)

    def evaluate(self, evidence: EventEvidence) -> list[ROIRuleTrigger]:
        """Evaluate all configured rules against an evidence object."""
        triggers: list[ROIRuleTrigger] = []
        for rule in self.rules:
            rule_type = str(rule.get("type", ""))
            if rule_type == "intrusion":
                triggers.append(self._intrusion(rule, evidence.objects))
            elif rule_type == "line_crossing":
                triggers.append(self._line_crossing(rule, evidence.objects))
            elif rule_type == "loitering":
                triggers.append(self._loitering(rule, evidence.objects))
            elif rule_type == "gathering":
                triggers.append(self._gathering(rule, evidence.objects))
            elif rule_type == "proximity":
                triggers.append(self._proximity(rule, evidence.objects))
        return triggers

    def _targets(self, rule: dict[str, Any], tracks: list[ObjectTrack]) -> list[ObjectTrack]:
        labels = set(rule.get("target_labels") or [])
        return [track for track in tracks if not labels or track.label in labels]

    def _roi_contains_track(self, roi_id: str | None, track: ObjectTrack) -> tuple[bool, float | None]:
        roi = self.rois.get(str(roi_id))
        if not roi:
            return False, None
        points = roi.get("points") or []
        for point in track.trajectory:
            if point_in_polygon([point.x, point.y], points):
                return True, point.timestamp
        return False, None

    def _trigger(self, rule: dict[str, Any], triggered: bool, tracks: list[int], trigger_time: list[float] | None = None) -> ROIRuleTrigger:
        return ROIRuleTrigger(
            rule_id=str(rule.get("rule_id", rule.get("type", "rule"))),
            rule_type=str(rule.get("type", "")),
            roi_id=rule.get("roi_id"),
            triggered=triggered,
            trigger_time=trigger_time,
            evidence_tracks=tracks,
            severity_hint=str(rule.get("severity", "none")),
        )

    def _intrusion(self, rule: dict[str, Any], tracks: list[ObjectTrack]) -> ROIRuleTrigger:
        matched: list[int] = []
        times: list[float] = []
        for track in self._targets(rule, tracks):
            ok, t = self._roi_contains_track(rule.get("roi_id"), track)
            if ok:
                matched.append(track.track_id)
                if t is not None:
                    times.append(t)
        return self._trigger(rule, bool(matched), matched, [min(times), min(times)] if times else None)

    def _line_crossing(self, rule: dict[str, Any], tracks: list[ObjectTrack]) -> ROIRuleTrigger:
        line = rule.get("line") or []
        if len(line) < 2:
            return self._trigger(rule, False, [])
        matched: list[int] = []
        times: list[float] = []
        for track in self._targets(rule, tracks):
            sides = [signed_line_side([p.x, p.y], line[0], line[1]) for p in track.trajectory]
            for idx in range(1, len(sides)):
                if sides[idx - 1] == 0 or sides[idx] == 0 or sides[idx - 1] * sides[idx] < 0:
                    matched.append(track.track_id)
                    times.append(track.trajectory[idx].timestamp)
                    break
        return self._trigger(rule, bool(matched), matched, [min(times), max(times)] if times else None)

    def _loitering(self, rule: dict[str, Any], tracks: list[ObjectTrack]) -> ROIRuleTrigger:
        threshold = float(rule.get("threshold_seconds", 8.0))
        matched: list[int] = []
        times: list[float] = []
        for track in self._targets(rule, tracks):
            roi_id = str(rule.get("roi_id"))
            roi_times = [p.timestamp for p in track.trajectory if point_in_polygon([p.x, p.y], self.rois.get(roi_id, {}).get("points") or [])]
            dwell = max(0.0, max(roi_times) - min(roi_times)) if len(roi_times) >= 2 else track.dwell_time
            if dwell >= threshold:
                matched.append(track.track_id)
                times.extend([min(roi_times) if roi_times else 0.0, max(roi_times) if roi_times else dwell])
        return self._trigger(rule, bool(matched), matched, [min(times), max(times)] if times else None)

    def _gathering(self, rule: dict[str, Any], tracks: list[ObjectTrack]) -> ROIRuleTrigger:
        threshold = int(rule.get("threshold_count", 3))
        roi_id = str(rule.get("roi_id"))
        counts: Counter[float] = Counter()
        track_by_time: defaultdict[float, list[int]] = defaultdict(list)
        for track in self._targets(rule, tracks):
            for point in track.trajectory:
                if point_in_polygon([point.x, point.y], self.rois.get(roi_id, {}).get("points") or []):
                    key = round(point.timestamp, 1)
                    counts[key] += 1
                    track_by_time[key].append(track.track_id)
        hit_times = [t for t, count in counts.items() if count >= threshold]
        matched = sorted({tid for t in hit_times for tid in track_by_time[t]})
        return self._trigger(rule, bool(hit_times), matched, [min(hit_times), max(hit_times)] if hit_times else None)

    def _proximity(self, rule: dict[str, Any], tracks: list[ObjectTrack]) -> ROIRuleTrigger:
        threshold = float(rule.get("threshold_pixels", 80.0))
        person_tracks = [t for t in tracks if t.label == "person"]
        other_labels = set(rule.get("object_labels") or ["equipment", "object", "vehicle"])
        other_tracks = [t for t in tracks if t.label in other_labels]
        matched: set[int] = set()
        hit_times: list[float] = []
        for person in person_tracks:
            for other in other_tracks:
                for pp in person.trajectory:
                    candidates = [op for op in other.trajectory if abs(op.timestamp - pp.timestamp) <= 0.5]
                    if any(distance([pp.x, pp.y], [op.x, op.y]) <= threshold for op in candidates):
                        matched.update({person.track_id, other.track_id})
                        hit_times.append(pp.timestamp)
                        break
        return self._trigger(rule, bool(matched), sorted(matched), [min(hit_times), max(hit_times)] if hit_times else None)

