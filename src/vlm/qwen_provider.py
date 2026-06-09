"""DashScope/Qwen provider adapter for STEAD review."""

from __future__ import annotations

import json
import os
from typing import Any

import requests
from dotenv import load_dotenv

from src.evidence.evidence_schema import EventEvidence
from src.vlm.json_validator import parse_vlm_review
from src.vlm.mock_provider import MockVLMProvider
from src.vlm.prompts import build_review_prompt
from src.vlm.review_provider import VLMReviewProvider
from src.vlm.vlm_schema import VLMReview

load_dotenv()


class QwenProvider(VLMReviewProvider):
    """Qwen/DashScope provider with mock fallback when no key is available."""

    def __init__(
        self,
        model: str = "qwen-vl-plus",
        api_key_env: str = "DASHSCOPE_API_KEY",
        timeout_sec: float = 20.0,
        mock_when_no_key: bool = True,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_sec = timeout_sec
        self.mock_when_no_key = mock_when_no_key
        self.mock = MockVLMProvider()

    def review(self, evidence: EventEvidence) -> VLMReview:
        """Call DashScope compatible API or fall back to mock review."""
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            if self.mock_when_no_key:
                return self.mock.review(evidence)
            raise RuntimeError(f"{self.api_key_env} is not configured")
        payload = self._payload(evidence)
        response = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "\n".join(str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item) for item in content)
        review, error = parse_vlm_review(str(content))
        if review is None:
            fallback = self.mock.review(evidence)
            fallback.reason = f"Qwen JSON invalid, fallback to mock: {error['message'][:120]}"
            return fallback
        return review

    def _payload(self, evidence: EventEvidence) -> dict[str, Any]:
        """Build DashScope compatible chat payload."""
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": build_review_prompt(evidence)}],
            "temperature": 0.0,
        }


def review_to_json(review: VLMReview) -> str:
    """Serialize a review for debugging or storage."""
    return json.dumps(review.model_dump(mode="json"), ensure_ascii=False, indent=2)

