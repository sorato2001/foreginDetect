"""Shared VLM execution helpers for image and video pipelines."""

from __future__ import annotations

import logging
from pathlib import Path

from src.evidence.evidence_schema import EventEvidence
from src.evidence.serializers import save_json
from src.vlm.provider_factory import build_vlm_provider, resolve_vlm_provider
from src.vlm.vlm_schema import VLMReview

logger = logging.getLogger("stead.pipeline")


def run_vlm_review(
    evidence: EventEvidence,
    output_dir: str | Path,
    vlm_provider: str,
    vlm_mode: str,
    timeout_sec: float,
    max_retries: int,
    fallback_on_error: bool,
    local_endpoint: str,
    local_model: str,
    local_max_images: int,
    local_evidence_mode: str = "minimal",
) -> tuple[VLMReview, str]:
    """Build the selected VLM provider, run review, and handle pipeline fallback."""
    output_path = Path(output_dir)
    effective_provider = resolve_vlm_provider(vlm_provider, vlm_mode)
    provider = build_vlm_provider(
        vlm_provider=effective_provider,
        vlm_mode=vlm_mode,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        fallback_on_error=fallback_on_error,
        artifact_dir=str(output_path),
        local_endpoint=local_endpoint,
        local_model=local_model,
        local_max_images=local_max_images,
        local_evidence_mode=local_evidence_mode,
    )
    try:
        return provider.review(evidence), effective_provider
    except Exception as exc:
        logger.exception("STEP 07 VLM: provider raised unexpectedly, using pipeline fallback")
        provider_name = _fallback_provider_name(effective_provider, vlm_mode)
        review = pipeline_fallback_review(exc, provider_name=provider_name)
        _write_pipeline_error(output_path, provider_name, exc)
        return review, effective_provider


def pipeline_fallback_review(exc: Exception, provider_name: str) -> VLMReview:
    """Last-resort fallback if a provider unexpectedly raises."""
    error_type = type(exc).__name__
    return VLMReview(
        is_anomaly=False,
        event_type="none",
        alarm_level_suggestion="none",
        confidence=0.0,
        evidence_time=[],
        evidence_tracks=[],
        matched_rules=[],
        reason=f"{provider_name} review failed in pipeline: {error_type}. Fallback review generated.",
        possible_false_alarm=True,
        recommended_action="manual review recommended",
        metadata={
            "provider": provider_name,
            "success": False,
            "error_type": error_type,
            "error_message": str(exc)[:500],
            "fallback": True,
            "pipeline_fallback": True,
        },
    )


def _fallback_provider_name(effective_provider: str, vlm_mode: str) -> str:
    if effective_provider == "gemma" or (vlm_mode == "local" and effective_provider != "mock"):
        return "local_gemma"
    return effective_provider


def _write_pipeline_error(output_dir: Path, provider_name: str, exc: Exception) -> None:
    if provider_name == "mock":
        return
    error_filename = "gemma_error.json" if provider_name == "local_gemma" else "qwen_error.json"
    save_json(
        {
            "provider": provider_name,
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
            "fallback": True,
            "pipeline_fallback": True,
        },
        str(output_dir / error_filename),
    )
