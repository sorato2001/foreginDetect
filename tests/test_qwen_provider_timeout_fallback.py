import os

import requests

from src.evidence.evidence_schema import EventEvidence
from src.vlm.qwen_provider import QwenProvider


def test_qwen_provider_timeout_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(requests, "post", raise_timeout)
    evidence = EventEvidence(event_id="e1", camera_id="cam01", video_path="x.mp4", time_range=[0, 1])
    review = QwenProvider(read_timeout_seconds=20, artifact_dir=str(tmp_path)).review(evidence)

    assert review.alarm_level_suggestion == "none"
    assert review.metadata["provider"] == "qwen"
    assert review.metadata["success"] is False
    assert review.metadata["fallback"] is True
    assert review.metadata["error_type"] == "ReadTimeout"
    assert "test-key" not in (tmp_path / "qwen_error.json").read_text(encoding="utf-8")

