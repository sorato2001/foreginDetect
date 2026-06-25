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


def test_pipeline_passes_sam_temporal_mode(monkeypatch, tmp_path):
    captured = {}

    class _CaptureSAMTrackingAdapter(_FakeSAMTrackingAdapter):
        def __init__(self, config):
            super().__init__(config)
            captured["temporal_mode"] = config.temporal_mode

    monkeypatch.setattr(analyze_event, "SAMTrackingAdapter", _CaptureSAMTrackingAdapter)

    analyze_event.run_pipeline(
        video_path="missing.mp4",
        camera_id="cam01",
        rules_path="configs/rules.example.yaml",
        output_dir=str(tmp_path / "sam_event_mode"),
        vlm_provider="mock",
        event_id="sam_event_mode",
        tracker="sam_tracking",
        sam_temporal_mode="faithful",
        save_visualization=False,
    )

    assert captured["temporal_mode"] == "faithful"


class _FakeSAMTrackingWithoutMaskAdapter:
    def __init__(self, config):
        self.config = config

    def analyze_video(self, video_path, max_frames=900):
        return SAMTrackingResult(
            tracks={7: [Detection("person", 0.91, [100, 100, 160, 220], 0, 0.0)]},
            fps=25.0,
            duration=1.0,
            frames=[
                SAMFrameResult(
                    frame_index=0,
                    timestamp=0.0,
                    detections=[{"label": "person", "confidence": 0.91, "bbox": [100, 100, 160, 220]}],
                    object_mask_count=1,
                    track_mask_available=False,
                    max_iou=0.0,
                    max_object_overlap=0.0,
                    suspicious=False,
                    window_count=0,
                    alarm=False,
                    track_mask_contours=[[[80, 220], [220, 220], [220, 320], [80, 320]]],
                )
            ],
            intrusion_events=[],
            metadata={"adapter": "sam_tracking", "degraded": True, "track_mask_seen": True, "processed_frames": 1},
        )


def test_sam_tracking_without_actual_track_mask_falls_back_to_config_rules(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze_event, "SAMTrackingAdapter", _FakeSAMTrackingWithoutMaskAdapter)
    output = tmp_path / "sam_no_mask_event"

    result = analyze_event.run_pipeline(
        video_path="missing.mp4",
        camera_id="cam01",
        rules_path="configs/rules.example.yaml",
        output_dir=str(output),
        vlm_provider="mock",
        event_id="sam_no_mask_event",
        tracker="sam_tracking",
        save_visualization=True,
    )

    evidence_text = Path(result["event_evidence"]).read_text(encoding="utf-8")
    summary_text = Path(result["visualization"]["summary_json"]).read_text(encoding="utf-8")
    assert '"rule_source": "config_rules"' in evidence_text
    assert '"sam_tracking_stage": null' in summary_text
    assert '"tracker_stage": {' in summary_text
