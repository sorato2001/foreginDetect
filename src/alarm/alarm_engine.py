"""Rule/VLM fusion engine for graded alarms."""

from __future__ import annotations

from src.alarm.alarm_schema import AlarmResult
from src.evidence.evidence_schema import EventEvidence, ROIRuleTrigger
from src.vlm.vlm_schema import VLMReview


LEVEL_WEIGHT = {"none": 0.0, "low": 0.4, "medium": 0.7, "high": 1.0}
RULE_WEIGHT = {"none": 0.0, "low": 0.3, "medium": 0.5, "high": 0.8}


class AlarmEngine:
    """Fuse structured rules and VLM review into final alarm decision."""

    def fuse(self, evidence: EventEvidence, review: VLMReview) -> AlarmResult:
        """Return final alarm result using the STEAD scoring formula."""
        triggered = [rule for rule in evidence.roi_rules if rule.triggered]
        rule_score = self._rule_score(triggered)
        vlm_score = float(review.confidence) * LEVEL_WEIGHT.get(review.alarm_level_suggestion, 0.0)
        final_score = 0.55 * rule_score + 0.45 * vlm_score
        if review.possible_false_alarm:
            final_score *= 0.6
        if not self._has_continuous_track(evidence):
            final_score *= 0.7
        final_score = max(0.0, min(1.0, final_score))
        final_level = self._level(final_score)
        is_alarm = final_level in {"medium", "high"}
        reasons = self._reasons(triggered, review, final_score)
        uncertainty = max(0.0, min(1.0, 1.0 - max(rule_score, vlm_score)))
        return AlarmResult(
            event_id=evidence.event_id,
            final_level=final_level,
            final_score=final_score,
            is_alarm=is_alarm,
            reasons=reasons,
            evidence_time=review.evidence_time,
            evidence_tracks=review.evidence_tracks,
            rule_score=rule_score,
            vlm_score=max(0.0, min(1.0, vlm_score)),
            uncertainty=uncertainty,
            action=self._action(final_level),
        )

    @staticmethod
    def _rule_score(rules: list[ROIRuleTrigger]) -> float:
        if not rules:
            return 0.0
        scores = [RULE_WEIGHT.get(rule.severity_hint, 0.0) for rule in rules]
        return min(1.0, max(scores) + 0.05 * max(0, len(rules) - 1))

    @staticmethod
    def _has_continuous_track(evidence: EventEvidence) -> bool:
        return any(len(track.trajectory) >= 2 or len(track.bboxes) >= 2 for track in evidence.objects)

    @staticmethod
    def _level(score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.50:
            return "medium"
        if score >= 0.30:
            return "low"
        return "none"

    @staticmethod
    def _action(level: str) -> str:
        return {
            "high": "immediate response",
            "medium": "manual verification",
            "low": "watch list",
            "none": "no action",
        }[level]

    @staticmethod
    def _reasons(rules: list[ROIRuleTrigger], review: VLMReview, final_score: float) -> list[str]:
        reasons = [f"Rule {rule.rule_id} triggered with {rule.severity_hint} severity." for rule in rules]
        reasons.append(f"VLM: {review.reason}")
        reasons.append(f"Final fused score={final_score:.3f}.")
        return reasons

