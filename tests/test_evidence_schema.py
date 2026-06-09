from src.evidence.evidence_schema import EventEvidence, FrameBBox, ObjectTrack, TrajectoryPoint


def test_event_evidence_serializes_round_trip():
    evidence = EventEvidence(
        event_id="e1",
        camera_id="cam01",
        video_path="demo.mp4",
        time_range=[0.0, 4.0],
        fps=25.0,
        objects=[
            ObjectTrack(
                track_id=1,
                label="person",
                confidence=0.9,
                bboxes=[FrameBBox(timestamp=0.0, bbox=[1, 2, 3, 4], confidence=0.9)],
                trajectory=[TrajectoryPoint(timestamp=0.0, x=2, y=3)],
            )
        ],
    )
    restored = EventEvidence.model_validate_json(evidence.model_dump_json())
    assert restored.event_id == "e1"
    assert restored.objects[0].label == "person"

