"""Factory helpers for selecting STEAD VLM review providers."""

from __future__ import annotations

from src.vlm.local_gemma_provider import LocalGemmaProvider
from src.vlm.mock_provider import MockVLMProvider
from src.vlm.qwen_provider import QwenProvider
from src.vlm.review_provider import VLMReviewProvider


def resolve_vlm_provider(vlm_provider: str = "auto", vlm_mode: str = "web") -> str:
    """Resolve VLM provider shorthand and reject conflicting mode/provider pairs."""
    provider_name = (vlm_provider or "auto").lower()
    mode = (vlm_mode or "web").lower()
    if provider_name == "auto":
        return "qwen" if mode == "web" else "gemma"
    if provider_name == "mock":
        return "mock"
    if provider_name == "qwen":
        if mode != "web":
            raise ValueError("--vlm-provider qwen requires --vlm-mode web")
        return "qwen"
    if provider_name in {"gemma", "local_gemma"}:
        if mode != "local":
            raise ValueError("--vlm-provider gemma requires --vlm-mode local")
        return "gemma"
    raise ValueError(f"Unsupported --vlm-provider: {vlm_provider}")


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
    mode = (vlm_mode or "web").lower()
    provider_name = resolve_vlm_provider(vlm_provider, mode)
    if provider_name == "mock":
        return MockVLMProvider()
    if provider_name == "gemma":
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
    return QwenProvider(
        timeout_sec=timeout_sec,
        read_timeout_seconds=timeout_sec,
        max_retries=max_retries,
        fallback_on_error=fallback_on_error,
        artifact_dir=artifact_dir,
    )
