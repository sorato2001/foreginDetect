from pathlib import Path

import cv2
import numpy as np

from src.pipeline.analyze_image import run_image_pipeline


def test_image_pipeline_with_mock_detections(tmp_path):
    image_path = tmp_path / "alarm.jpg"
    cv2.imwrite(str(image_path), np.full((480, 640, 3), 245, dtype=np.uint8))
    output = tmp_path / "image_event"

    result = run_image_pipeline(
        image_path=str(image_path),
        camera_id="cam_img",
        rules_path="configs/rules.example.yaml",
        output_dir=str(output),
        vlm_provider="mock",
        event_id="image_event",
        mock_detections=True,
    )

    assert result["input_type"] == "image"
    assert Path(result["event_evidence"]).exists()
    assert Path(result["vlm_review"]).exists()
    assert Path(result["alarm_result"]).exists()
    assert Path(result["pipeline_log"]).exists()
    assert Path(result["visualization"]["annotated_image"]).exists()
    assert "annotated_video" not in result["visualization"]
    evidence_text = Path(result["event_evidence"]).read_text(encoding="utf-8")
    assert "input_image" in evidence_text
