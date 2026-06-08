"""YOLO detector wrapper for railway perimeter security events.

The detector is intentionally defensive: if ultralytics, torch, or the model file is
unavailable, it returns an empty detection list instead of crashing the monitoring
process. This keeps the demo runnable on machines without GPU/model assets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

COCO_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
}

ANIMAL_CLASSES = {"cat", "dog", "horse", "cow", "sheep"}
DEFAULT_TARGET_CLASSES = {
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "dog",
    "cat",
    "horse",
    "cow",
    "sheep",
    "animal",
}


@dataclass(slots=True)
class Detection:
    """Single object detection result in image coordinates."""

    class_name: str
    confidence: float
    bbox: list[float]
    center: list[float]
    raw_class_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize detection to JSON-friendly dict."""
        data = {
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "center": [round(float(v), 2) for v in self.center],
        }
        if self.raw_class_name and self.raw_class_name != self.class_name:
            data["raw_class_name"] = self.raw_class_name
        return data


class YoloDetector:
    """Thin wrapper around ultralytics YOLO with COCO target filtering."""

    _model_cache: dict[str, Any] = {}

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        target_classes: Iterable[str] | None = None,
    ) -> None:
        self.model_path = model_path or "yolov8n.pt"
        self.conf_threshold = float(conf_threshold)
        self.target_classes = set(target_classes or DEFAULT_TARGET_CLASSES)
        self.model = self._load_model(self.model_path)

    @classmethod
    def _load_model(cls, model_path: str) -> Any | None:
        """Load and cache a YOLO model; return None on failure."""
        cache_key = str(Path(model_path))
        if cache_key in cls._model_cache:
            return cls._model_cache[cache_key]
        try:
            from ultralytics import YOLO
            import torch

            original_load = torch.load

            def patched_load(file_obj: Any, *args: Any, **kwargs: Any) -> Any:
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return original_load(file_obj, *args, **kwargs)

            torch.load = patched_load
            try:
                model = YOLO(model_path)
            finally:
                torch.load = original_load
            cls._model_cache[cache_key] = model
            logger.info("Loaded YOLO model for railway security: %s", model_path)
            return model
        except Exception as exc:  # pragma: no cover - depends on runtime assets
            logger.warning("YOLO model unavailable (%s): %s", model_path, exc)
            cls._model_cache[cache_key] = None
            return None

    @staticmethod
    def _normalize_class(class_name: str) -> str:
        """Merge animal-like COCO classes into the high-level animal label."""
        if class_name in ANIMAL_CLASSES:
            return "animal"
        return class_name

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        """Run YOLO detection on one image and return JSON-friendly results."""
        if self.model is None:
            return []
        try:
            results = self.model.predict(
                source=image_path,
                conf=self.conf_threshold,
                imgsz=640,
                verbose=False,
            )
            if not results:
                return []

            model_names = getattr(self.model, "names", {}) or COCO_NAMES
            detections: list[dict[str, Any]] = []
            result = results[0]
            if result.boxes is None:
                return []

            for box in result.boxes:
                cls_id = int(box.cls.item())
                raw_name = str(model_names.get(cls_id, COCO_NAMES.get(cls_id, cls_id)))
                normalized_name = self._normalize_class(raw_name)
                if raw_name not in self.target_classes and normalized_name not in self.target_classes:
                    continue
                conf = float(box.conf.item())
                if conf < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                det = Detection(
                    class_name=normalized_name,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    center=[(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    raw_class_name=raw_name,
                )
                detections.append(det.to_dict())
            return detections
        except Exception as exc:  # pragma: no cover - depends on image/model runtime
            logger.error("YOLO detection failed for %s: %s", image_path, exc, exc_info=True)
            return []

