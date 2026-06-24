"""DashScope/Qwen provider adapter for STEAD review."""

from __future__ import annotations

import json
import logging
import base64
import mimetypes
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from pydantic import ValidationError

from src.evidence.evidence_schema import EventEvidence
from src.evidence.serializers import save_json
from src.vlm.json_validator import parse_vlm_review
from src.vlm.prompts import build_review_prompt, summarize_evidence
from src.vlm.review_provider import VLMReviewProvider
from src.vlm.vlm_schema import VLMReview

load_dotenv()
logger = logging.getLogger(__name__)

QWEN_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.HTTPError,
    requests.exceptions.RequestException,
    json.JSONDecodeError,
    ValueError,
    ValidationError,
)


class QwenProvider(VLMReviewProvider):
    """Qwen/DashScope provider with bounded retry and safe fallback."""

    def __init__(
        self,
        model: str = "qwen-vl-plus",
        api_key_env: str = "DASHSCOPE_API_KEY",
        timeout_sec: float = 20.0,
        connect_timeout_seconds: float | None = None,
        read_timeout_seconds: float | None = None,
        max_retries: int = 0,
        retry_backoff_seconds: float = 2.0,
        fallback_on_error: bool = True,
        mock_when_no_key: bool = True,
        artifact_dir: str | None = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_sec = float(timeout_sec)
        self.connect_timeout_seconds = float(connect_timeout_seconds if connect_timeout_seconds is not None else min(10.0, self.timeout_sec))
        self.read_timeout_seconds = float(read_timeout_seconds if read_timeout_seconds is not None else self.timeout_sec)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.fallback_on_error = bool(fallback_on_error)
        self.mock_when_no_key = mock_when_no_key
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None

    def review(self, evidence: EventEvidence) -> VLMReview:
        """Call DashScope compatible API or return a valid fallback review."""
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return self._fallback_review(evidence, "MissingAPIKey", f"{self.api_key_env} is not configured", retry_count=0)

        vlm_input = self._vlm_input_metadata(evidence)
        payload = self._payload(evidence)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(
                    "Qwen review attempt %s/%s: model=%s connect_timeout=%.1fs read_timeout=%.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    self.model,
                    self.connect_timeout_seconds,
                    self.read_timeout_seconds,
                )
                response = requests.post(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=(self.connect_timeout_seconds, self.read_timeout_seconds),
                )
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                self._write_raw_response(data)
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "\n".join(
                        str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                review, error = parse_vlm_review(str(content))
                if review is None:
                    raise ValueError(error["message"] if error else "invalid Qwen JSON")
                review.metadata.update(self._base_metadata(success=True, retry_count=attempt, fallback=False))
                review.metadata["vlm_input"] = vlm_input
                logger.info(
                    "Qwen review attempt %s/%s: success level=%s confidence=%.3f normalized=%s",
                    attempt + 1,
                    self.max_retries + 1,
                    review.alarm_level_suggestion,
                    review.confidence,
                    review.metadata.get("normalized", False),
                )
                return review
            except QWEN_EXCEPTIONS as exc:
                last_exc = exc
                logger.warning(
                    "Qwen review attempt %s/%s failed: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    type(exc).__name__,
                )
                if attempt < self.max_retries:
                    logger.info(
                        "Qwen review retry scheduled: sleep=%.1fs next_attempt=%s",
                        self.retry_backoff_seconds * (attempt + 1),
                        attempt + 2,
                    )
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                self._write_error(exc, retry_count=attempt)
                fallback = self._fallback_review(evidence, type(exc).__name__, str(exc), retry_count=attempt)
                fallback.metadata["vlm_input"] = vlm_input
                return fallback

        assert last_exc is not None
        fallback = self._fallback_review(evidence, type(last_exc).__name__, str(last_exc), retry_count=self.max_retries)
        fallback.metadata["vlm_input"] = vlm_input
        return fallback

    def _payload(self, evidence: EventEvidence) -> dict[str, Any]:
        """Build DashScope compatible chat payload."""
        prompt = build_review_prompt(evidence)
        image_url = self._first_keyframe_data_url(evidence)
        content: str | list[dict[str, Any]]
        if image_url:
            content = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]
            logger.info("Qwen payload: using multimodal image+text review")
        else:
            content = prompt
            logger.info("Qwen payload: using text-only structured evidence review")
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
        }

    def _vlm_input_metadata(self, evidence: EventEvidence) -> dict[str, Any]:
        """Return the non-secret evidence packet sent to Qwen."""
        prompt_text = build_review_prompt(evidence)
        image_path = self._first_keyframe_path(evidence)
        return {
            "provider": "qwen",
            "evidence_mode": "qwen",
            "image_count": 1 if image_path else 0,
            "image_paths": [image_path] if image_path else [],
            "prompt_text": prompt_text,
            "prompt_text_chars": len(prompt_text),
            "evidence_summary": summarize_evidence(evidence),
        }

    @staticmethod
    def _first_keyframe_data_url(evidence: EventEvidence) -> str | None:
        """Return the first local keyframe encoded as a data URL."""
        image_path = QwenProvider._first_keyframe_path(evidence)
        if image_path:
            path = Path(image_path)
            mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{data}"
        return None

    @staticmethod
    def _first_keyframe_path(evidence: EventEvidence) -> str | None:
        """Return the first existing local keyframe path."""
        for keyframe in evidence.keyframes:
            path = Path(keyframe.frame_path)
            if path.exists() and path.is_file():
                return keyframe.frame_path
        return None

    def _base_metadata(self, success: bool, retry_count: int, fallback: bool) -> dict[str, Any]:
        """Return metadata without secrets."""
        return {
            "provider": "qwen",
            "success": success,
            "timeout_seconds": self.timeout_sec,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "read_timeout_seconds": self.read_timeout_seconds,
            "retry_count": retry_count,
            "max_retries": self.max_retries,
            "fallback": fallback,
            "model": self.model,
        }

    def _fallback_review(self, evidence: EventEvidence, error_type: str, error_message: str, retry_count: int) -> VLMReview:
        """Build a valid fallback VLMReview after Qwen failure."""
        triggered = [rule for rule in evidence.roi_rules if rule.triggered]
        return VLMReview(
            is_anomaly=False,
            event_type="none",
            alarm_level_suggestion="none",
            confidence=0.0,
            evidence_time=[],
            evidence_tracks=[],
            matched_rules=[rule.rule_id for rule in triggered],
            reason=f"Qwen review failed: {error_type} after {self.read_timeout_seconds:g}s. Fallback review generated.",
            possible_false_alarm=True,
            recommended_action="manual review recommended",
            metadata={
                **self._base_metadata(success=False, retry_count=retry_count, fallback=True),
                "error_type": error_type,
                "error_message": self._sanitize_error(error_message),
            },
        )

    def _write_raw_response(self, data: dict[str, Any]) -> None:
        """Save raw Qwen response without request headers or API key."""
        if not self.artifact_dir:
            return
        logger.info("Qwen raw response artifact: %s", self.artifact_dir / "qwen_raw_response.json")
        save_json(
            {"provider": "qwen", "success": True, "timestamp": self._now(), "response": data},
            str(self.artifact_dir / "qwen_raw_response.json"),
        )

    def _write_error(self, exc: Exception, retry_count: int) -> None:
        """Save Qwen error metadata without secrets."""
        if not self.artifact_dir:
            return
        logger.info("Qwen error artifact: %s", self.artifact_dir / "qwen_error.json")
        save_json(
            {
                **self._base_metadata(success=False, retry_count=retry_count, fallback=True),
                "error_type": type(exc).__name__,
                "error_message": self._sanitize_error(str(exc)),
                "timestamp": self._now(),
            },
            str(self.artifact_dir / "qwen_error.json"),
        )

    @staticmethod
    def _now() -> str:
        """Return current UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _sanitize_error(message: str) -> str:
        """Keep error messages short and avoid leaking bearer tokens."""
        return message.replace("Bearer ", "Bearer [REDACTED] ")[:500]


def review_to_json(review: VLMReview) -> str:
    """Serialize a review for debugging or storage."""
    return json.dumps(review.model_dump(mode="json"), ensure_ascii=False, indent=2)
