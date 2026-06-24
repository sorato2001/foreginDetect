from pathlib import Path

from src.pipeline import analyze_event


class BrokenProvider:
    def review(self, evidence):
        raise RuntimeError("provider exploded")


def test_pipeline_vlm_failure_no_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze_event, "build_vlm_provider", lambda **kwargs: BrokenProvider())
    output = tmp_path / "event"
    result = analyze_event.run_pipeline(
        video_path="missing.mp4",
        camera_id="cam01",
        rules_path="configs/rules.example.yaml",
        output_dir=str(output),
        vlm_provider="qwen",
        mock_detections=True,
    )

    assert Path(result["event_evidence"]).exists()
    assert Path(result["vlm_review"]).exists()
    assert Path(result["alarm_result"]).exists()
