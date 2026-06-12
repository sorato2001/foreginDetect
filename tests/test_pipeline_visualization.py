from pathlib import Path

from src.pipeline.analyze_event import run_pipeline


def test_pipeline_writes_visualization(tmp_path):
    output = tmp_path / "event_vis"
    result = run_pipeline(
        video_path="missing.mp4",
        camera_id="cam01",
        rules_path="configs/rules.example.yaml",
        output_dir=str(output),
        vlm_provider="mock",
        event_id="event_vis",
        mock_detections=True,
    )

    overview = result["visualization"]["overview_image"]
    animation = result["visualization"]["evidence_animation"]
    summary = result["visualization"]["summary_json"]
    assert overview is not None
    assert animation is not None
    assert Path(overview).exists()
    assert Path(animation).exists()
    assert Path(summary).exists()
