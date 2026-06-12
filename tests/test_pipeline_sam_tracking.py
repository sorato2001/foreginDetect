from pathlib import Path

from src.perception.detector import Detection
from src.perception.sam_tracking_adapter import SAMFrameResult, SAMTrackingResult
from src.pipeline import analyze_event


class _FakeSAMTrackingAdapter:
    def __init__(self, config):
        self.config = config

    def analyze_video(self, video_path, max_frames=900):
        return SAMTrackingResult(
            tracks={
                7: [
                    Detection("person", 0.91, [100, 100, 160, 220], 0, 0.0),
                    Detection("person", 0.92, [120, 110, 180, 230], 1, 1.0),
                ]
            },
            fps=25.0,
            duration=2.0,
            frames=[
                SAMFrameResult(
                    frame_index=0,
                    timestamp=0.0,
                    detections=[],
                    object_mask_count=1,
                    track_mask_available=True,
                    max_iou=0.2,
                    max_object_overlap=0.6,
                    suspicious=True,
                    window_count=1,
                    alarm=False,
                )
            ],
            intrusion_events=[],
            metadata={"adapter": "sam_tracking", "degraded": False, "processed_frames": 1},
        )


def test_pipeline_writes_sam_tracking_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze_event, "SAMTrackingAdapter", _FakeSAMTrackingAdapter)
    output = tmp_path / "sam_event"

    result = analyze_event.run_pipeline(
        video_path="missing.mp4",
        camera_id="cam01",
        rules_path="configs/rules.example.yaml",
        output_dir=str(output),
        vlm_provider="mock",
        event_id="sam_event",
        tracker="sam_tracking",
        save_visualization=False,
    )

    artifact = result["sam_tracking_result"]
    assert artifact is not None
    assert Path(artifact).exists()
    evidence_text = Path(result["event_evidence"]).read_text(encoding="utf-8")
    assert "sam_tracking" in evidence_text
