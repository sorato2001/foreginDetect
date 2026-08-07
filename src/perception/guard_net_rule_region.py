"""GroundingDINO + SAM2 guard-net rule-region generation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.perception.open_vocab_detector import (
    GroundingDINOOpenVocabularyDetector,
    OpenVocabularyDetection,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GuardNetConfig:
    backend: str = "grounding_dino"
    text_prompt: str = "black chain link fence. black metal mesh fence. wire mesh fence. protective fence."
    model_path: str = "weights/groundingdino_swint_ogc.pth"
    config_path: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path: str = "weights/groundingdino_swint_ogc.pth"
    sam2_config: str | None = None
    sam2_checkpoint: str | None = None
    box_threshold: float = 0.10
    text_threshold: float = 0.10
    nms_threshold: float = 0.50
    max_box_area_ratio: float = 0.60
    scan_frames: int = 30
    sample_every: int = 5
    band_side_fraction: float = 0.08
    band_top_padding: int = 0
    band_bottom_padding: int = 0
    band_horizontal_padding: int = 0
    export_yolo_seg: bool = False
    yolo_class_id: int = 0
    output_dir: str | None = None
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "detector_score": 0.25,
            "sam_score": 0.20,
            "horizontal_span_score": 0.20,
            "column_coverage_score": 0.15,
            "boundary_consistency_score": 0.15,
            "area_sanity_score": 0.05,
        }
    )


@dataclass(slots=True)
class GuardNetCandidate:
    raw_mask: Any
    detector_score: float
    sam_score: float
    bbox: list[float]
    phrase: str
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class GuardNetResult:
    raw_mask: Any | None = None
    mask_all: Any | None = None
    refined_mask: Any | None = None
    continuous_mask: Any | None = None
    candidate_masks: list[Any] = field(default_factory=list)
    candidate_boxes: list[list[float]] = field(default_factory=list)
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    selected_candidate: int | None = None
    continuous_band_geometry: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.continuous_mask is not None

    @property
    def mask(self) -> Any | None:
        return self.continuous_mask


class GuardNetRuleRegionGenerator:
    """Generate a static guard-net mask without owning person tracking state."""

    def __init__(
        self,
        config: GuardNetConfig | None = None,
        device: str = "cuda",
        detector: Any | None = None,
        predictor_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config or GuardNetConfig()
        self.device = device
        self._detector = detector
        self._predictor_factory = predictor_factory
        self._predictor: Any | None = None

    def generate(self, frame: Any) -> GuardNetResult:
        try:
            import numpy as np
        except Exception as exc:
            return GuardNetResult(error=f"numpy_unavailable: {exc}")
        try:
            logger.info("GuardNet GroundingDINO detection start: backend=%s", self.config.backend)
            detections = self._get_detector().detect(frame, self.config.text_prompt)
            detections = self._filter_detections(detections, frame.shape[1], frame.shape[0])
            logger.info("GuardNet GroundingDINO detection done: boxes=%s", len(detections))
            if not detections:
                return GuardNetResult(metadata={"backend": self.config.backend}, error="no_guard_net_boxes")
            candidates = self._segment_candidate_boxes(frame, detections)
            if not candidates:
                return GuardNetResult(candidate_boxes=[item.bbox for item in detections], error="sam2_no_valid_mask")
            mask_all = np.logical_or.reduce([np.asarray(item.raw_mask).astype(bool) for item in candidates])
            refined_mask, continuous_mask, geometry, kept_components = build_refined_component_mask(mask_all)
            if refined_mask is None:
                return GuardNetResult(
                    mask_all=mask_all,
                    candidate_masks=[item.raw_mask for item in candidates],
                    candidate_boxes=[item.bbox for item in detections],
                    candidate_scores=[self._candidate_json(item) for item in candidates],
                    metadata={"backend": self.config.backend, "aggregation_mode": "mask_all"},
                    error="mask_all_no_connected_component",
                )
            if continuous_mask is None:
                return GuardNetResult(
                    raw_mask=refined_mask,
                    mask_all=mask_all,
                    refined_mask=refined_mask,
                    candidate_masks=[item.raw_mask for item in candidates],
                    candidate_boxes=[item.bbox for item in detections],
                    candidate_scores=[self._candidate_json(item) for item in candidates],
                    metadata={"backend": self.config.backend, "aggregation_mode": "mask_all_component_polygons", "kept_components": kept_components},
                    error="continuous_band_invalid",
                )
            result = GuardNetResult(
                raw_mask=refined_mask,
                mask_all=mask_all,
                refined_mask=refined_mask,
                continuous_mask=continuous_mask,
                candidate_masks=[item.raw_mask for item in candidates],
                candidate_boxes=[item.bbox for item in detections],
                candidate_scores=[self._candidate_json(item) for item in candidates],
                selected_candidate=None,
                continuous_band_geometry=geometry,
                metadata={
                    "enabled": True,
                    "backend": self.config.backend,
                    "text_prompt": self.config.text_prompt,
                    "source_frame_index": 0,
                    "candidate_count": len(candidates),
                    "aggregation_mode": "mask_all_components",
                    "component_count": int(geometry.get("component_count", 0)),
                    "mask_all_area": int(mask_all.sum()),
                    "refined_mask_area": int(refined_mask.sum()),
                    "kept_components": kept_components,
                    "aggregate_confidence": max((item.detector_score * item.sam_score for item in candidates), default=0.0),
                    "continuous_band": geometry,
                    "error": None,
                },
            )
            self._save_artifacts(result, frame)
            return result
        except Exception as exc:
            logger.warning("GuardNet generation failed: %s", exc)
            return GuardNetResult(error=f"guard_net_generation_failed: {type(exc).__name__}: {exc}")

    def _segment_candidate_boxes(
        self,
        frame: Any,
        detections: list[OpenVocabularyDetection],
    ) -> list[GuardNetCandidate]:
        """Segment every detector box; candidate selection happens only after mask_all."""
        import numpy as np

        predictor = self._get_predictor()
        predictor.set_image(np.asarray(frame)[:, :, ::-1].copy())
        candidates: list[GuardNetCandidate] = []
        for detection in detections:
            logger.info("GuardNet SAM2 box segmentation start: box=%s", [round(value, 1) for value in detection.bbox])
            masks, scores = predictor.predict(
                box=np.asarray(detection.bbox, dtype=np.float32),
                multimask_output=True,
            )[:2]
            mask_values = np.asarray(masks)
            if mask_values.ndim == 2:
                mask_values = mask_values[None, ...]
            score_values = np.asarray(scores).reshape(-1) if scores is not None else np.ones(len(mask_values))
            for index, mask in enumerate(mask_values):
                binary = np.asarray(mask).squeeze().astype(bool)
                if binary.ndim != 2 or not binary.any():
                    continue
                candidates.append(
                    GuardNetCandidate(
                        raw_mask=binary,
                        detector_score=float(detection.score),
                        sam_score=float(score_values[min(index, len(score_values) - 1)]),
                        bbox=detection.bbox,
                        phrase=detection.phrase,
                    )
                )
            logger.info("GuardNet SAM2 box segmentation done: accumulated_masks=%s", len(candidates))
        return candidates

    def aggregate_results(self, results: list[GuardNetResult], frame: Any) -> GuardNetResult:
        """Merge sampled-frame mask_all results instead of selecting a best frame."""
        import numpy as np

        usable = [item for item in results if item.mask_all is not None]
        if not usable:
            return GuardNetResult(error="guard_net_scan_no_valid_mask_all")
        mask_all = np.logical_or.reduce([np.asarray(item.mask_all).astype(bool) for item in usable])
        refined_mask, continuous_mask, geometry, kept_components = build_refined_component_mask(mask_all)
        if refined_mask is None:
            return GuardNetResult(mask_all=mask_all, error="mask_all_no_connected_component")
        if continuous_mask is None:
            return GuardNetResult(mask_all=mask_all, refined_mask=refined_mask, error="continuous_band_invalid")
        result = GuardNetResult(
            raw_mask=refined_mask,
            mask_all=mask_all,
            refined_mask=refined_mask,
            continuous_mask=continuous_mask,
            candidate_masks=[mask for item in usable for mask in item.candidate_masks],
            candidate_boxes=[box for item in usable for box in item.candidate_boxes],
            candidate_scores=[score for item in usable for score in item.candidate_scores],
            continuous_band_geometry=geometry,
            metadata={
                "enabled": True,
                "backend": self.config.backend,
                "text_prompt": self.config.text_prompt,
                "source_frame_indices": [item.metadata.get("source_frame_index") for item in usable],
                "candidate_count": sum(len(item.candidate_scores) for item in usable),
                "aggregation_mode": "multi_frame_mask_all_components",
                "component_count": int(geometry.get("component_count", 0)),
                "mask_all_area": int(mask_all.sum()),
                "refined_mask_area": int(refined_mask.sum()),
                "kept_components": kept_components,
                "aggregate_confidence": max((float(item.metadata.get("aggregate_confidence", 0.0)) for item in usable), default=0.0),
                "continuous_band": geometry,
                "error": None,
            },
        )
        self._save_artifacts(result, frame)
        return result

    def _get_detector(self) -> Any:
        if self._detector is None:
            if self.config.backend != "grounding_dino":
                raise RuntimeError(f"unsupported_guard_net_backend: {self.config.backend}")
            self._detector = GroundingDINOOpenVocabularyDetector(
                config_path=self.config.config_path,
                checkpoint_path=self.config.checkpoint_path or self.config.model_path,
                device=self.device,
                box_threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
            )
        return self._detector

    def _get_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        if self._predictor_factory is not None:
            self._predictor = self._predictor_factory()
            return self._predictor
        try:
            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from src.perception.sam_tracking_adapter import _resolve_sam2_config
        except Exception as exc:
            raise RuntimeError("sam2_image_predictor_unavailable") from exc
        if not self.config.sam2_config or not self.config.sam2_checkpoint:
            raise RuntimeError("sam2_image_predictor_unavailable: --sam2-config and --sam2-checkpoint are required")
        config_dir, config_name = _resolve_sam2_config(self.config.sam2_config)
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            model = build_sam2(config_name, self.config.sam2_checkpoint, device=self.device)
        self._predictor = SAM2ImagePredictor(model)
        return self._predictor

    def _filter_detections(self, detections: list[OpenVocabularyDetection], width: int, height: int) -> list[OpenVocabularyDetection]:
        clipped: list[OpenVocabularyDetection] = []
        for item in detections:
            x1, y1, x2, y2 = item.bbox
            pad_x = max(0.0, x2 - x1) * 0.03
            pad_y = max(0.0, y2 - y1) * 0.03
            x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y
            box = [max(0.0, min(float(width - 1), x1)), max(0.0, min(float(height - 1), y1)), max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2))]
            area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / max(1.0, float(width * height))
            if box[2] > box[0] and box[3] > box[1] and area_ratio <= self.config.max_box_area_ratio:
                clipped.append(OpenVocabularyDetection(box, item.score, item.phrase))
        return _nms(clipped, self.config.nms_threshold)

    def _score_candidates(self, candidates: list[GuardNetCandidate], width: int, height: int) -> None:
        for candidate in candidates:
            breakdown = reliability_score(
                candidate.raw_mask,
                candidate.detector_score,
                candidate.sam_score,
                width,
                height,
                weights=self.config.score_weights,
            )
            candidate.score_breakdown = breakdown
            candidate.score = sum(breakdown.values())

    @staticmethod
    def _candidate_json(candidate: GuardNetCandidate) -> dict[str, Any]:
        return {
            "bbox": candidate.bbox,
            "phrase": candidate.phrase,
            "detector_score": candidate.detector_score,
            "sam_score": candidate.sam_score,
            "mask_area": int(candidate.raw_mask.sum()),
        }

    def _save_artifacts(self, result: GuardNetResult, frame: Any) -> None:
        if not self.config.output_dir:
            return
        try:
            import cv2
            import numpy as np
            output = Path(self.config.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            result.metadata.update(
                {
                    "raw_mask_artifact": str(output / "fence_refined.png"),
                    "mask_all_artifact": str(output / "mask_all.png"),
                    "refined_mask_artifact": str(output / "fence_refined.png"),
                    "rule_region_artifact": str(output / "fence_refined.png"),
                    "continuous_mask_artifact": str(output / "fence_refined.png"),
                    "overlay_artifact": str(output / "rule_region_overlay.jpg"),
                    "detection_boxes_artifact": str(output / "detection_boxes.jpg"),
                    "sam_candidates_artifact": str(output / "sam_candidates.jpg"),
                    "yolo_seg_artifact": None,
                }
            )
            detection_boxes = np.asarray(frame).copy()
            for box in result.candidate_boxes:
                x1, y1, x2, y2 = [int(value) for value in box]
                cv2.rectangle(detection_boxes, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.imwrite(str(output / "detection_boxes.jpg"), detection_boxes)
            candidate_view = detection_boxes.copy()
            if result.mask_all is not None:
                raw = np.asarray(result.mask_all).astype(bool)
                raw_tint = np.zeros_like(candidate_view)
                raw_tint[:, :, 2] = 230
                candidate_view[raw] = (0.55 * candidate_view[raw] + 0.45 * raw_tint[raw]).astype(np.uint8)
            cv2.imwrite(str(output / "sam_candidates.jpg"), candidate_view)
            for index, candidate_mask in enumerate(result.candidate_masks):
                cv2.imwrite(
                    str(output / f"candidate_mask_{index:03d}.png"),
                    np.asarray(candidate_mask).astype(np.uint8) * 255,
                )
            if result.mask_all is not None:
                cv2.imwrite(str(output / "mask_all.png"), (np.asarray(result.mask_all).astype(np.uint8) * 255))
            if result.refined_mask is not None:
                cv2.imwrite(str(output / "fence_refined.png"), (np.asarray(result.refined_mask).astype(np.uint8) * 255))
            overlay = np.asarray(frame).copy()
            if result.refined_mask is not None:
                refined = np.asarray(result.refined_mask).astype(bool)
                refined_tint = np.zeros_like(overlay)
                refined_tint[:, :, 1] = 144
                refined_tint[:, :, 2] = 255
                overlay[refined] = (0.70 * overlay[refined] + 0.30 * refined_tint[refined]).astype(np.uint8)
            geometry = result.continuous_band_geometry or {}
            for box in result.candidate_boxes:
                x1, y1, x2, y2 = [int(value) for value in box]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.imwrite(str(output / "rule_region_overlay.jpg"), overlay)
            (output / "guard_net_result.json").write_text(json.dumps(self._json_metadata(result), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("GuardNet artifact save failed: %s", exc)

    @staticmethod
    def _json_metadata(result: GuardNetResult) -> dict[str, Any]:
        return {
            "candidate_boxes": result.candidate_boxes,
            "candidate_scores": result.candidate_scores,
            "selected_candidate": result.selected_candidate,
            "mask_all_area": int(result.mask_all.sum()) if result.mask_all is not None else 0,
            "refined_mask_area": int(result.refined_mask.sum()) if result.refined_mask is not None else 0,
            "continuous_band": result.continuous_band_geometry,
            "metadata": result.metadata,
            "error": result.error,
        }


def _nms(detections: list[OpenVocabularyDetection], threshold: float) -> list[OpenVocabularyDetection]:
    kept: list[OpenVocabularyDetection] = []
    for item in sorted(detections, key=lambda value: value.score, reverse=True):
        if all(_iou(item.bbox, other.bbox) <= threshold for other in kept):
            kept.append(item)
    return kept


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b - intersection else 0.0


def reliability_score(
    mask: Any,
    detector_score: float,
    sam_score: float,
    width: int,
    height: int,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return weighted, inspectable reliability components for one raw mask."""
    import numpy as np

    binary = np.asarray(mask).astype(bool)
    ys, xs = np.where(binary)
    if not len(xs):
        return {"detector_score": 0.0, "sam_score": 0.0, "horizontal_span_score": 0.0, "column_coverage_score": 0.0, "boundary_consistency_score": 0.0, "area_sanity_score": 0.0}
    span = float(xs.max() - xs.min() + 1) / max(1, width)
    columns = np.unique(xs)
    coverage = float(len(columns)) / max(1, xs.max() - xs.min() + 1)
    top: list[float] = []
    bottom: list[float] = []
    for x in columns:
        values = ys[xs == x]
        top.append(float(values.min()))
        bottom.append(float(values.max()))
    if len(columns) >= 2:
        residual = np.std(np.asarray(top) - np.polyval(np.polyfit(columns, top, 1), columns)) + np.std(np.asarray(bottom) - np.polyval(np.polyfit(columns, bottom, 1), columns))
    else:
        residual = float(height)
    boundary = max(0.0, 1.0 - float(residual) / max(1.0, height * 0.25))
    ratio = float(binary.sum()) / max(1, width * height)
    area = 1.0 if 0.0005 <= ratio <= 0.75 else 0.0
    weights = weights or GuardNetConfig().score_weights
    components = {
        "detector_score": max(0.0, min(1.0, detector_score)),
        "sam_score": max(0.0, min(1.0, sam_score)),
        "horizontal_span_score": max(0.0, min(1.0, span)),
        "column_coverage_score": coverage,
        "boundary_consistency_score": boundary,
        "area_sanity_score": area,
    }
    return {name: float(weights.get(name, 0.0)) * value for name, value in components.items()}


def largest_mask_components(
    mask_all: Any,
    keep_components: int = 1,
    min_area_ratio: float = 0.02,
) -> tuple[Any | None, list[dict[str, Any]]]:
    """Remove mask_all noise and retain its largest connected regular region."""
    import cv2
    import numpy as np

    binary = np.asarray(mask_all).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return None, []
    ranked = sorted(
        ((index, int(stats[index, cv2.CC_STAT_AREA])) for index in range(1, count)),
        key=lambda item: item[1],
        reverse=True,
    )
    largest_area = ranked[0][1]
    refined = np.zeros_like(binary)
    kept: list[dict[str, Any]] = []
    for rank, (label, area) in enumerate(ranked):
        if rank >= max(1, keep_components) or area < largest_area * min_area_ratio:
            break
        refined[labels == label] = 1
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        kept.append(
            {
                "component": rank,
                "area": area,
                "area_ratio": float(area / max(1, largest_area)),
                "bbox": [x, y, x + width, y + height],
            }
        )
    return (refined if refined.any() else None), kept


def build_refined_component_mask(
    mask_all: Any,
    min_area_ratio: float = 0.01,
    min_area: int = 64,
    max_components: int = 12,
) -> tuple[Any | None, Any | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """Keep substantial components and return the refined SAM mask directly."""
    import cv2
    import numpy as np

    binary = np.asarray(mask_all).astype(bool)
    if binary.ndim != 2 or not binary.any():
        return None, None, None, []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    ranked = sorted(
        ((index, int(stats[index, cv2.CC_STAT_AREA])) for index in range(1, count)),
        key=lambda item: item[1], reverse=True,
    )
    if not ranked:
        return None, None, None, []
    largest_area = ranked[0][1]
    refined = np.zeros(binary.shape, dtype=np.uint8)
    kept: list[dict[str, Any]] = []
    for rank, (label, area) in enumerate(ranked[:max(1, max_components)]):
        if area < max(min_area, int(round(largest_area * min_area_ratio))):
            continue
        component = (labels == label).astype(np.uint8)
        refined[component > 0] = 1
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        item = {
            "component": int(rank),
            "area": int(area),
            "area_ratio": float(area / max(1, largest_area)),
            "bbox": [x, y, x + width, y + height],
            "raw_area": int(area),
        }
        kept.append({key: item[key] for key in ("component", "area", "area_ratio", "bbox")})
    if not kept:
        return None, None, None, []
    geometry = {
        "mode": "mask_all_components",
        "component_count": len(kept),
        "raw_mask_area": int(refined.sum()),
        "continuous_mask_area": int(refined.sum()),
        "rule_region_area": int(refined.sum()),
    }
    continuous = refined.copy()
    return refined.astype(bool), continuous.astype(bool), geometry, kept


def _boundary_points(mask: Any, min_height_ratio: float = 0.05) -> tuple[Any, Any, Any]:
    """Extract robust per-column upper and lower envelopes."""
    import numpy as np

    binary = np.asarray(mask).astype(bool)
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("empty refined mask")
    width = binary.shape[1]
    tops = np.full(width, np.iinfo(np.int32).max, dtype=np.int64)
    bottoms = np.full(width, -1, dtype=np.int64)
    np.minimum.at(tops, xs, ys)
    np.maximum.at(bottoms, xs, ys)
    valid = bottoms >= tops
    columns = np.where(valid)[0]
    top_values = tops[valid]
    bottom_values = bottoms[valid]
    heights = bottom_values - top_values
    keep = heights >= max(2.0, float(heights.max()) * min_height_ratio)
    return columns[keep], top_values[keep], bottom_values[keep]


def _fit_line_ransac(xs: Any, ys: Any, threshold: float, iterations: int = 300) -> tuple[float, float, int]:
    """Fit y=m*x+b deterministically and reject jagged mask-boundary outliers."""
    import numpy as np

    x_values = np.asarray(xs, dtype=np.float64)
    y_values = np.asarray(ys, dtype=np.float64)
    if len(x_values) < 2:
        raise ValueError("not enough boundary points")
    rng = np.random.default_rng(0)
    best_model: tuple[float, float] | None = None
    best_inliers = np.zeros(len(x_values), dtype=bool)
    for _ in range(iterations):
        first, second = rng.choice(len(x_values), 2, replace=False)
        delta = x_values[second] - x_values[first]
        if abs(delta) < 1e-9:
            continue
        slope = (y_values[second] - y_values[first]) / delta
        intercept = y_values[first] - slope * x_values[first]
        inliers = np.abs(y_values - (slope * x_values + intercept)) <= threshold
        if int(inliers.sum()) > int(best_inliers.sum()):
            best_model = (float(slope), float(intercept))
            best_inliers = inliers
    if best_model is None or int(best_inliers.sum()) < 2:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        return float(slope), float(intercept), len(x_values)
    slope, intercept = np.polyfit(x_values[best_inliers], y_values[best_inliers], 1)
    return float(slope), float(intercept), int(best_inliers.sum())


def build_fitted_line_band(
    mask: Any,
    top_padding: int = 0,
    bottom_padding: int = 0,
    horizontal_padding: int = 0,
    line_tolerance: float = 0.05,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Fit robust top/bottom lines to a refined mask and fill the enclosed band."""
    import cv2
    import numpy as np

    binary = np.asarray(mask).astype(bool)
    try:
        xs, tops, bottoms = _boundary_points(binary)
    except ValueError:
        return None, None
    if len(xs) < 2 or int(xs.max()) <= int(xs.min()):
        return None, None
    vertical_extent = max(1.0, float(bottoms.max() - tops.min()))
    threshold = max(2.0, vertical_extent * max(0.001, line_tolerance))
    try:
        top_slope, top_intercept, top_inliers = _fit_line_ransac(xs, tops, threshold)
        bottom_slope, bottom_intercept, bottom_inliers = _fit_line_ransac(xs, bottoms, threshold)
    except ValueError:
        return None, None

    height, width = binary.shape[:2]
    x_min = max(0, int(xs.min()) - max(0, horizontal_padding))
    x_max = min(width - 1, int(xs.max()) + max(0, horizontal_padding))

    def point(slope: float, intercept: float, x: int, padding: int) -> list[int]:
        y = int(round(slope * x + intercept)) + padding
        return [x, max(0, min(height - 1, y))]

    left_top = point(top_slope, top_intercept, x_min, -max(0, top_padding))
    right_top = point(top_slope, top_intercept, x_max, -max(0, top_padding))
    left_bottom = point(bottom_slope, bottom_intercept, x_min, max(0, bottom_padding))
    right_bottom = point(bottom_slope, bottom_intercept, x_max, max(0, bottom_padding))
    if left_bottom[1] <= left_top[1] or right_bottom[1] <= right_top[1]:
        return None, None
    polygon = [left_top, right_top, right_bottom, left_bottom]
    continuous = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(continuous, [np.asarray(polygon, dtype=np.int32)], 1)
    geometry = {
        "mode": "mask_all_largest_component_ransac_lines",
        "left_top": left_top,
        "right_top": right_top,
        "left_bottom": left_bottom,
        "right_bottom": right_bottom,
        "top_line": {
            "slope": top_slope,
            "intercept": top_intercept - max(0, top_padding),
            "inliers": top_inliers,
            "total": int(len(xs)),
        },
        "bottom_line": {
            "slope": bottom_slope,
            "intercept": bottom_intercept + max(0, bottom_padding),
            "inliers": bottom_inliers,
            "total": int(len(xs)),
        },
        "line_tolerance_px": threshold,
        "polygon": polygon,
        "raw_mask_area": int(binary.sum()),
        "continuous_mask_area": int(continuous.sum()),
        "x_min": x_min,
        "x_max": x_max,
    }
    return continuous, geometry


def build_continuous_band(mask: Any, side_fraction: float = 0.08, top_padding: int = 0, bottom_padding: int = 0, horizontal_padding: int = 0) -> tuple[Any | None, dict[str, Any] | None]:
    """Fill a complete endpoint-defined quadrilateral, including internal occlusions."""
    import cv2
    import numpy as np

    binary = np.asarray(mask).astype(bool)
    ys, xs = np.where(binary)
    if not len(xs) or xs.max() <= xs.min():
        return None, None
    x_min, x_max = int(xs.min()), int(xs.max())
    side_window = max(3, int((x_max - x_min + 1) * max(0.01, min(0.5, side_fraction))))
    points: dict[str, list[int]] = {}
    occupied_columns = np.unique(xs)
    side_count = min(len(occupied_columns), max(1, side_window))
    for name, columns in (("left", occupied_columns[:side_count]), ("right", occupied_columns[-side_count:])):
        tops = [int(np.where(binary[:, x])[0].min()) for x in columns]
        bottoms = [int(np.where(binary[:, x])[0].max()) for x in columns]
        if not tops or not bottoms:
            return None, None
        points[f"{name}_top"] = [x_min if name == "left" else x_max, int(np.median(tops))]
        points[f"{name}_bottom"] = [x_min if name == "left" else x_max, int(np.median(bottoms))]
    lt, rt = points["left_top"], points["right_top"]
    lb, rb = points["left_bottom"], points["right_bottom"]
    if lb[1] <= lt[1] or rb[1] <= rt[1]:
        return None, None
    height, width = binary.shape[:2]
    polygon = [[max(0, lt[0] - horizontal_padding), max(0, lt[1] - top_padding)], [min(width - 1, rt[0] + horizontal_padding), max(0, rt[1] - top_padding)], [min(width - 1, rb[0] + horizontal_padding), min(height - 1, rb[1] + bottom_padding)], [max(0, lb[0] - horizontal_padding), min(height - 1, lb[1] + bottom_padding)]]
    continuous = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(continuous, [np.asarray(polygon, dtype=np.int32)], 1)
    top_slope = (rt[1] - lt[1]) / max(1, rt[0] - lt[0])
    bottom_slope = (rb[1] - lb[1]) / max(1, rb[0] - lb[0])
    geometry = {"mode": "endpoint_lines", "left_top": lt, "right_top": rt, "left_bottom": lb, "right_bottom": rb, "top_line": {"slope": top_slope, "intercept": lt[1] - top_slope * lt[0]}, "bottom_line": {"slope": bottom_slope, "intercept": lb[1] - bottom_slope * lb[0]}, "polygon": polygon, "raw_mask_area": int(binary.sum()), "continuous_mask_area": int(continuous.sum()), "x_min": x_min, "x_max": x_max}
    return continuous, geometry
