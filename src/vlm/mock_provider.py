"""Deterministic offline VLM provider for tests and demos."""

from __future__ import annotations

from src.evidence.evidence_schema import EventEvidence
from src.vlm.review_provider import VLMReviewProvider
from src.vlm.vlm_schema import VLMReview


class MockVLMProvider(VLMReviewProvider):
    """Review evidence using simple deterministic rules without external API."""

    def review(self, evidence: EventEvidence) -> VLMReview:
        """Return a plausible VLMReview based on triggered rule severity."""
        triggered = [rule for rule in evidence.roi_rules if rule.triggered]
        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        best = max(triggered, key=lambda r: severity_order.get(r.severity_hint, 0), default=None)
        level = best.severity_hint if best else "none"
        is_anomaly = level in {"medium", "high"}
        tracks = sorted({track for rule in triggered for track in rule.evidence_tracks})
        evidence_time = [rule.trigger_time for rule in triggered if rule.trigger_time]
        reason = (
            f"Mock review: matched {len(triggered)} structured rule trigger(s)."
            if triggered
            else "Mock review: no sustained ROI or trajectory anomaly evidence."
        )
        return VLMReview(
            is_anomaly=is_anomaly,
            event_type=best.rule_type if best else "none",
            alarm_level_suggestion=level,
            confidence=0.78 if is_anomaly else 0.55,
            evidence_time=evidence_time,
            evidence_tracks=tracks,
            matched_rules=[rule.rule_id for rule in triggered],
            reason=reason,
            possible_false_alarm=not is_anomaly,
            recommended_action="notify security staff and save the event clip" if is_anomaly else "no action required",
        )

