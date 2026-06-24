import requests

from src.evidence.evidence_schema import EventEvidence
from src.vlm.qwen_provider import QwenProvider


def test_vlm_review_metadata(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(requests, "post", raise_timeout)
    evidence = EventEvidence(event_id="e1", camera_id="cam01", video_path="x.mp4", time_range=[0, 1])
    review = QwenProvider().review(evidence)

    assert review.metadata["fallback"] is True
    assert review.metadata["provider"] == "qwen"
    assert review.metadata["success"] is False
    assert review.metadata["vlm_input"]["provider"] == "qwen"
    assert review.metadata["vlm_input"]["evidence_mode"] == "qwen"
    assert "event_evidence_summary" in review.metadata["vlm_input"]["prompt_text"]
