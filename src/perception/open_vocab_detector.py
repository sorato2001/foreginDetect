"""Lazy open-vocabulary detection backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OpenVocabularyDetection:
    """A detector candidate in image-space xyxy coordinates."""

    bbox: list[float]
    score: float
    phrase: str


class OpenVocabularyDetector:
    """Backend interface for open-vocabulary detectors."""

    def detect(self, frame: Any, text_prompt: str) -> list[OpenVocabularyDetection]:
        raise NotImplementedError


class GroundingDINOOpenVocabularyDetector(OpenVocabularyDetector):
    """GroundingDINO backend with all heavy imports deferred until inference."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        model: Any | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model = model

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.config_path.exists():
            raise FileNotFoundError(f"GroundingDINO config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {self.checkpoint_path}")
        try:
            from groundingdino.util.inference import load_model
        except Exception as exc:
            raise RuntimeError("groundingdino_unavailable: install GroundingDINO separately") from exc
        self._model = load_model(str(self.config_path), str(self.checkpoint_path), device=self.device)
        return self._model

    def detect(self, frame: Any, text_prompt: str) -> list[OpenVocabularyDetection]:
        try:
            import numpy as np
            from PIL import Image
            import groundingdino.datasets.transforms as T
            from groundingdino.util.inference import predict
        except Exception as exc:
            raise RuntimeError("groundingdino_unavailable: missing inference dependencies") from exc

        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("GroundingDINO expects an HxWx3 image")
        image_pil = Image.fromarray(image[:, :, ::-1].copy())
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        transformed, _ = transform(image_pil, None)
        boxes, logits, phrases = predict(
            model=self._load_model(),
            image=transformed,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        height, width = image.shape[:2]
        result: list[OpenVocabularyDetection] = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = [float(value) for value in box.tolist()]
            result.append(
                OpenVocabularyDetection(
                    bbox=[(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height],
                    score=float(logit.max().item() if hasattr(logit, "max") else logit),
                    phrase=str(phrase),
                )
            )
        return result
