from pathlib import Path

from src.pipeline.analyze_event import run_pipeline


def test_pipeline_smoke_with_mock_detections(tmp_path):
    output = tmp_path / "event_001"
    result = run_pipeline(
        video_path="missing.mp4",
        camera_id="cam01",
        rules_path="configs/rules.example.yaml",
        output_dir=str(output),
        vlm_provider="mock",
        event_id="event_001",
        mock_detections=True,
    )
    assert result["is_alarm"] is True
    assert Path(result["event_evidence"]).exists()
    assert Path(result["vlm_review"]).exists()
    assert Path(result["alarm_result"]).exists()
    log_path = Path(result["pipeline_log"])
    assert log_path.exists()
    assert "STEP 02 detector/tracker" in log_path.read_text(encoding="utf-8")
