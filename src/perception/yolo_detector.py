"""YOLO detector adapter with a no-crash fallback."""

from __future__ import annotations

import logging
from typing import Any

from src.perception.detector import Detection

logger = logging.getLogger(__name__)


class YoloDetector:
    """Ultralytics YOLO adapter used by the research pipeline."""

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.35) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model: Any | None = None
        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            logger.info("Loaded STEAD YOLO detector: %s", model_path)
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.warning("STEAD YOLO unavailable, using empty detections: %s", exc)

    def detect_frame(self, frame: Any, frame_index: int, timestamp: float) -> list[Detection]:
        """Detect objects in a frame; return empty list when model is unavailable."""
        if self.model is None:
            return []
        results = self.model.predict(source=frame, conf=self.conf_threshold, imgsz=640, verbose=False)
        detections: list[Detection] = []
        if not results or results[0].boxes is None:
            return detections
        names = getattr(self.model, "names", {}) or {}
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            label = str(names.get(cls_id, cls_id))
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    label=label,
                    confidence=float(box.conf.item()),
                    bbox=[x1, y1, x2, y2],
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
            )
        return detections

