"""Factory helpers for selecting STEAD VLM review providers."""

from __future__ import annotations

from src.vlm.local_gemma_provider import LocalGemmaProvider
from src.vlm.mock_provider import MockVLMProvider
from src.vlm.qwen_provider import QwenProvider
from src.vlm.review_provider import VLMReviewProvider


def build_vlm_provider(
    vlm_provider: str = "mock",
    vlm_mode: str = "web",
    timeout_sec: float = 20.0,
    max_retries: int = 0,
    fallback_on_error: bool = True,
    artifact_dir: str | None = None,
    local_endpoint: str = "http://localhost:8082/v1/chat/completions",
    local_model: str = "gemma-4-26B",
    local_max_images: int = 1,
    local_evidence_mode: str = "minimal",
) -> VLMReviewProvider:
    """Build a VLM provider from an explicit, non-conflicting mode/provider pair."""
    provider_name = (vlm_provider or "mock").lower()
    mode = (vlm_mode or "web").lower()
    if provider_name == "auto":
        provider_name = "qwen" if mode == "web" else "gemma"
    if provider_name == "mock":
        return MockVLMProvider()
    if provider_name in {"gemma", "local_gemma"}:
        if mode != "local":
            raise ValueError("Gemma/local_gemma provider requires vlm_mode='local'")
        return LocalGemmaProvider(
            endpoint=local_endpoint,
            model=local_model,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            fallback_on_error=fallback_on_error,
            artifact_dir=artifact_dir,
            max_images=local_max_images,
            evidence_mode=local_evidence_mode,
        )
    if provider_name != "qwen":
        raise ValueError(f"Unsupported VLM provider: {vlm_provider}")
    if mode != "web":
        raise ValueError("Qwen provider requires vlm_mode='web'")
    return QwenProvider(
        timeout_sec=timeout_sec,
        read_timeout_seconds=timeout_sec,
        max_retries=max_retries,
        fallback_on_error=fallback_on_error,
        artifact_dir=artifact_dir,
    )
