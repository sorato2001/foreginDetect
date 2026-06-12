import requests

from src.evidence.evidence_schema import EventEvidence
from src.vlm.qwen_provider import QwenProvider


def test_qwen_provider_connection_error_fallback(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    def raise_connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("network down")

    monkeypatch.setattr(requests, "post", raise_connection_error)
    evidence = EventEvidence(event_id="e1", camera_id="cam01", video_path="x.mp4", time_range=[0, 1])
    review = QwenProvider(max_retries=1, retry_backoff_seconds=0).review(evidence)

    assert review.metadata["error_type"] == "ConnectionError"
    assert review.metadata["retry_count"] == 1
    assert review.metadata["fallback"] is True

