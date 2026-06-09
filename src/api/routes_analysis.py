"""Analysis routes for the STEAD API."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from src.pipeline.analyze_event import run_pipeline

router = APIRouter()


class AnalyzeEventRequest(BaseModel):
    """Request body for event analysis."""

    video_path: str
    camera_id: str
    rules_config: str = "configs/rules.example.yaml"
    output: str = "outputs/api_event"
    vlm_provider: str = "mock"


@router.post("/analysis/event")
def analyze_event(request: AnalyzeEventRequest) -> dict:
    """Run the STEAD event analysis pipeline."""
    result = run_pipeline(
        video_path=request.video_path,
        camera_id=request.camera_id,
        rules_path=request.rules_config,
        output_dir=request.output,
        vlm_provider=request.vlm_provider,
    )
    return {"status": "completed", **result}


@router.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "stead"}

