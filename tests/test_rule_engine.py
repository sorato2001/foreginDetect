from src.evidence.evidence_schema import EventEvidence, ObjectTrack, TrajectoryPoint
from src.rules.rule_engine import RuleEngine


def _evidence(track):
    return EventEvidence(event_id="e1", camera_id="cam01", video_path="x.mp4", time_range=[0, 5], objects=[track])


def test_intrusion_loitering_and_line_crossing():
    config = {
        "rois": [{"roi_id": "r1", "type": "polygon", "points": [[100, 100], [500, 100], [500, 400], [100, 400]]}],
        "rules": [
            {"rule_id": "intrude", "type": "intrusion", "roi_id": "r1", "target_labels": ["person"], "severity": "high"},
            {"rule_id": "cross", "type": "line_crossing", "line": [[160, 0], [160, 500]], "target_labels": ["person"], "severity": "high"},
            {"rule_id": "loiter", "type": "loitering", "roi_id": "r1", "target_labels": ["person"], "threshold_seconds": 3, "severity": "medium"},
        ],
    }
    track = ObjectTrack(
        track_id=1,
        label="person",
        confidence=0.9,
        trajectory=[
            TrajectoryPoint(timestamp=0, x=120, y=120),
            TrajectoryPoint(timestamp=2, x=180, y=150),
            TrajectoryPoint(timestamp=4, x=260, y=180),
        ],
    )
    triggers = RuleEngine(config).evaluate(_evidence(track))
    triggered = {item.rule_id for item in triggers if item.triggered}
    assert {"intrude", "cross", "loiter"} <= triggered

