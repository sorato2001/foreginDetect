"""Configuration schemas for the STEAD prototype."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model and provider settings."""

    detector: str = "yolo"
    yolo_model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.35
    tracker: str = "simple_iou"
    vlm_provider: str = "mock"
    qwen_model: str = "qwen-vl-plus"
    dashscope_api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_sec: float = 20.0


class PipelineConfig(BaseModel):
    """Runtime pipeline settings."""

    camera_id: str = "cam01"
    output_dir: str = "outputs/demo"
    max_keyframes: int = Field(default=5, ge=1, le=10)
    window_size: float = 4.0
    stride: float = 2.0

