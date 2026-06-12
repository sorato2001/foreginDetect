import cv2
import numpy as np

from src.evidence.evidence_schema import EventEvidence, KeyframeInfo
from src.vlm.qwen_provider import QwenProvider


def test_qwen_payload_uses_keyframe_image(tmp_path):
    image_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(image_path), np.full((32, 32, 3), 255, dtype=np.uint8))
    evidence = EventEvidence(
        event_id="e1",
        camera_id="cam01",
        video_path=str(image_path),
        time_range=[0, 1],
        keyframes=[KeyframeInfo(timestamp=0, frame_path=str(image_path), reason="input_image")],
    )

    payload = QwenProvider()._payload(evidence)
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/")
    assert content[1]["type"] == "text"

