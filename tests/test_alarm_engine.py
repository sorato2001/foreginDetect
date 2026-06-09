from src.alarm.alarm_engine import AlarmEngine
from src.evidence.evidence_schema import EventEvidence, ObjectTrack, ROIRuleTrigger, TrajectoryPoint
from src.vlm.vlm_schema import VLMReview


def test_alarm_engine_high_fusion():
    evidence = EventEvidence(
        event_id="e1",
        camera_id="cam01",
        video_path="x.mp4",
        time_range=[0, 4],
        roi_rules=[
            ROIRuleTrigger(
                rule_id="intrusion",
                rule_type="intrusion",
                roi_id="r1",
                triggered=True,
                trigger_time=[1, 1],
                evidence_tracks=[1],
                severity_hint="high",
            )
        ],
        objects=[
            ObjectTrack(
                track_id=1,
                label="person",
                trajectory=[TrajectoryPoint(timestamp=0, x=1, y=1), TrajectoryPoint(timestamp=2, x=2, y=2)],
            )
        ],
    )
    review = VLMReview(
        is_anomaly=True,
        event_type="intrusion",
        alarm_level_suggestion="high",
        confidence=0.9,
        evidence_time=[[1, 2]],
        evidence_tracks=[1],
        matched_rules=["intrusion"],
        reason="confirmed intrusion",
        possible_false_alarm=False,
        recommended_action="respond",
    )
    alarm = AlarmEngine().fuse(evidence, review)
    assert alarm.final_level == "high"
    assert alarm.is_alarm is True

