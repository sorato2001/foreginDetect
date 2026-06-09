"""Provider interface for VLM/LLM event review."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.evidence.evidence_schema import EventEvidence
from src.vlm.vlm_schema import VLMReview


class VLMReviewProvider(ABC):
    """Unified interface for multimodal review providers."""

    @abstractmethod
    def review(self, evidence: EventEvidence) -> VLMReview:
        """Review structured evidence and return a validated result."""

