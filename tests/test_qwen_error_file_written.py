import json

import requests

from src.evidence.evidence_schema import EventEvidence
from src.vlm.qwen_provider import QwenProvider


def test_qwen_error_file_written(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    def raise_timeout(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(requests, "post", raise_timeout)
    evidence = EventEvidence(event_id="e1", camera_id="cam01", video_path="x.mp4", time_range=[0, 1])
    QwenProvider(artifact_dir=str(tmp_path)).review(evidence)

    error_path = tmp_path / "qwen_error.json"
    assert error_path.exists()
    data = json.loads(error_path.read_text(encoding="utf-8"))
    assert data["provider"] == "qwen"
    assert data["success"] is False
    assert data["error_type"] == "ReadTimeout"
    assert "test-key" not in json.dumps(data)

