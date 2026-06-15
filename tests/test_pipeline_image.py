from pathlib import Path

import cv2
import numpy as np

from src.perception.detector import Detection
from src.perception.sam_tracking_adapter import SAMFrameResult, SAMTrackingResult
from src.pipeline.analyze_image import run_image_pipeline
from src.pipeline.analyze_image import run_image_batch_pipeline
from src.pipeline import analyze_image


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


class _FakeImageSAMTrackingAdapter:
    def __init__(self, config):
        self.config = config

    def analyze_image(self, image_path):
        return SAMTrackingResult(
            tracks={1: [Detection("person", 0.93, [100, 100, 170, 260], 0, 0.0)]},
            fps=0.0,
            duration=1.0,
            frames=[
                SAMFrameResult(
                    frame_index=0,
                    timestamp=0.0,
                    detections=[{"label": "person", "confidence": 0.93, "bbox": [100, 100, 170, 260], "frame_index": 0, "timestamp": 0.0}],
                    object_mask_count=1,
                    track_mask_available=True,
                    max_iou=0.08,
                    max_object_overlap=0.76,
                    suspicious=True,
                    window_count=1,
                    alarm=True,
                    track_mask_contours=[[[80, 220], [220, 220], [220, 320], [80, 320]]],
                    object_mask_contours=[[[[100, 100], [170, 100], [170, 260], [100, 260]]]],
                )
            ],
            intrusion_events=[],
            metadata={"adapter": "sam_tracking", "degraded": False, "track_mask_seen": True, "detections_seen": True, "processed_frames": 1},
        )


def test_image_pipeline_can_use_sam_tracking_rule(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze_image, "SAMTrackingAdapter", _FakeImageSAMTrackingAdapter)
    image_path = tmp_path / "alarm.jpg"
    cv2.imwrite(str(image_path), np.full((480, 640, 3), 245, dtype=np.uint8))
    output = tmp_path / "image_sam_event"

    result = run_image_pipeline(
        image_path=str(image_path),
        camera_id="cam_img",
        rules_path="configs/rules.example.yaml",
        output_dir=str(output),
        vlm_provider="mock",
        event_id="image_sam_event",
        tracker="sam_tracking",
        save_visualization=True,
    )

    assert result["sam_tracking_result"] is not None
    assert Path(result["sam_tracking_result"]).exists()
    assert Path(result["visualization"]["annotated_image"]).exists()
    evidence_text = Path(result["event_evidence"]).read_text(encoding="utf-8")
    assert "sam_tracking_mask_iou" in evidence_text
    assert "sam_tracking" in evidence_text


def test_image_batch_pipeline_writes_summary(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ["a.jpg", "b.png"]:
        cv2.imwrite(str(image_dir / name), np.full((240, 320, 3), 245, dtype=np.uint8))
    output = tmp_path / "batch"

    result = run_image_batch_pipeline(
        image_path=str(image_dir),
        camera_id="cam_img",
        rules_path="configs/rules.example.yaml",
        output_dir=str(output),
        vlm_provider="mock",
        mock_detections=True,
    )

    assert result["input_type"] == "image_batch"
    assert result["total"] == 2
    assert result["succeeded"] == 2
    assert Path(result["summary_json"]).exists()
    assert all(Path(item["event_evidence"]).exists() for item in result["results"])
