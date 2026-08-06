"""Guard-net/fence rule-region adapter for STEAD SAMTracking.

This adapter reuses the standalone GroundingDINO + SAM2 fence segmentation
script when it is available. It produces a single binary rule-region mask that
SAMTracking can feed into the existing mask-IoU/object-overlap/sliding-window
judgment path.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_GUARD_NET_PROMPT = (
    "entire continuous black metal chain link fence"
)


@dataclass(slots=True)
class GuardNetRegionConfig:
    """Configuration for GroundingDINO + SAM2 guard-net rule mask generation."""

    text_prompt: str = DEFAULT_GUARD_NET_PROMPT
    box_threshold: float = 0.15
    text_threshold: float = 0.15
    max_box_area_ratio: float = 0.6
    candidate_count: int = 12
    crop_roi: str | None = None
    selection_mode: str = "best"
    target_area_ratio: float = 0.55
    mask_output_mode: str = "sam"
    continuous_band_margin: int = 4
    continuous_band_endpoint_source: str = "largest-component"
    save_selected_sam_mask: bool = False
    groundingdino_config: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    groundingdino_checkpoint: str = "weights/groundingdino_swint_ogc.pth"
    groundingdino_repo: str | None = None
    sam2_repo: str = "../demo/sam2-main"


@dataclass(slots=True)
class GuardNetRegionResult:
    """Guard-net rule region output for SAMTracking."""

    mask: Any | None
    contours: list
    source: str = "guard_net"
    available: bool = False
    selected_candidate_index: int | None = None
    candidate_count: int = 0
    selection_score: float | None = None
    mask_output_mode: str = "sam"
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata without embedding the full mask array."""
        data = asdict(self)
        data.pop("mask", None)
        return data


class GuardNetRegionAdapter:
    """Generate a guard-net/fence rule-region mask from a reference frame."""

    def __init__(
        self,
        config: GuardNetRegionConfig | None = None,
        sam2_config: str | None = None,
        sam2_checkpoint: str | None = None,
        device: str = "cuda",
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or GuardNetRegionConfig()
        self.sam2_config = sam2_config
        self.sam2_checkpoint = sam2_checkpoint
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None

    def analyze_frame(self, frame_bgr: Any, frame_index: int = 0) -> GuardNetRegionResult:
        """Run GroundingDINO + SAM2 on one BGR frame and return a rule mask."""
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            return self._unavailable(f"opencv_numpy_unavailable: {type(exc).__name__}: {exc}")

        if frame_bgr is None:
            return self._unavailable("empty_frame")
        try:
            fence = self._load_fence_module()
        except Exception as exc:
            return self._unavailable(f"fence_grounded_sam_unavailable: {type(exc).__name__}: {exc}")

        output_dir = self.output_dir or Path("outputs") / "guard_net"
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_path = output_dir / f"reference_frame_{frame_index}.jpg"
        cv2.imwrite(str(frame_path), frame_bgr)

        try:
            script_dir = Path(fence.__file__).resolve().parent
            device = fence.resolve_device(self.device)
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            height, width = frame_bgr.shape[:2]
            roi = fence.parse_crop_roi(self.config.crop_roi, width, height)
            x1, y1, x2, y2 = roi
            crop_rgb = image_rgb[y1:y2, x1:x2].copy()

            grounding_config = self._resolve_external_path(self.config.groundingdino_config, script_dir)
            grounding_checkpoint = self._resolve_external_path(self.config.groundingdino_checkpoint, script_dir)
            dino_model = fence.load_groundingdino_model(grounding_config, grounding_checkpoint, device)
            boxes, scores, phrases = fence.run_groundingdino(
                dino_model,
                crop_rgb,
                self.config.text_prompt,
                self.config.box_threshold,
                self.config.text_threshold,
                device,
            )
            detections = fence.filter_detections(
                boxes,
                scores,
                phrases,
                roi_origin=(x1, y1),
                crop_shape=crop_rgb.shape[:2],
                filter_large_box=True,
                max_box_area_ratio=self.config.max_box_area_ratio,
            )
            selected = fence.select_final_detections(
                detections,
                selection_mode=self.config.selection_mode,
                target_area_ratio=self.config.target_area_ratio,
            )
            if not selected:
                return self._unavailable(
                    "no_valid_guard_net_candidate",
                    metadata={"candidate_count": len(detections), "detections": [asdict(item) for item in detections]},
                )

            sam2_repo = self._resolve_external_path(self.config.sam2_repo, script_dir)
            sam2_config = self._resolve_sam2_config(fence, sam2_repo)
            if self.sam2_checkpoint:
                sam2_checkpoint = self._resolve_external_path(self.sam2_checkpoint, script_dir)
            else:
                sam2_checkpoint = self._resolve_external_path("../demo/sam2-main/checkpoints/sam2_hiera_large.pt", script_dir)
            predictor = fence.load_sam2_predictor(sam2_repo, sam2_config, sam2_checkpoint, device)
            masks = fence.run_sam2_on_boxes(
                predictor,
                crop_rgb,
                detections,
                roi,
                (height, width),
                output_dir,
                cv2,
            )
            final_dets = [det for det in selected if det.selected and det.mask_path]
            if not masks or not final_dets:
                return self._unavailable(
                    "sam2_returned_no_guard_net_mask",
                    metadata={"candidate_count": len(detections), "detections": [asdict(item) for item in detections]},
                )
            mask = np.asarray(masks[0]).astype(np.uint8)
            selected_det = final_dets[0]
            all_mask_path = output_dir / "mask_all.png"
            cv2.imwrite(str(all_mask_path), mask * 255)
            fence.draw_visualization(frame_bgr, detections, [mask], output_dir / "fence_grounded_sam_vis.jpg", cv2)
            metadata = {
                "enabled": True,
                "groundingdino_loaded": True,
                "sam2_image_predictor_loaded": True,
                "text_prompt": self.config.text_prompt,
                "box_threshold": self.config.box_threshold,
                "text_threshold": self.config.text_threshold,
                "max_box_area_ratio": self.config.max_box_area_ratio,
                "candidate_count": len(detections),
                "candidate_count_limit": self.config.candidate_count,
                "selection_mode": self.config.selection_mode,
                "target_area_ratio": self.config.target_area_ratio,
                "selected_candidate_index": selected_det.index,
                "selection_score": selected_det.reliability_score,
                "mask_output_mode": "sam",
                "requested_mask_output_mode": self.config.mask_output_mode,
                "continuous_band": {
                    "margin": self.config.continuous_band_margin,
                    "endpoint_source": self.config.continuous_band_endpoint_source,
                    "available": False,
                    "fallback_reason": "fence_grounded_sam_1_outputs_raw_sam_masks",
                },
                "raw_selected_sam_mask_saved": self.config.save_selected_sam_mask,
                "fallback_reason": None,
                "artifacts": {
                    "output_dir": str(output_dir),
                    "reference_frame": str(frame_path),
                    "groundingdino_config": str(grounding_config),
                    "groundingdino_checkpoint": str(grounding_checkpoint),
                    "sam2_repo": str(sam2_repo),
                    "sam2_config": str(sam2_config),
                    "sam2_checkpoint": str(sam2_checkpoint),
                    "mask_0": selected_det.mask_path,
                    "mask_all": str(all_mask_path),
                    "sam_mask_0": selected_det.mask_path,
                    "visualization": str(output_dir / "fence_grounded_sam_vis.jpg"),
                },
                "detections": [asdict(item) for item in detections],
            }
            return GuardNetRegionResult(
                mask=mask,
                contours=self._mask_to_contours(mask),
                available=bool(mask.sum() > 0),
                selected_candidate_index=selected_det.index,
                candidate_count=len(detections),
                selection_score=float(selected_det.reliability_score),
                mask_output_mode="sam",
                metadata=metadata,
            )
        except Exception as exc:
            return self._unavailable(f"{type(exc).__name__}: {exc}")

    def _load_fence_module(self) -> Any:
        repo = self.config.groundingdino_repo
        candidates: list[Path] = []
        if repo:
            candidates.append(Path(repo))
        current = Path(__file__).resolve()
        candidates.extend(
            [
                current.parents[4] / "GroundingDINO",
                current.parents[3] / "GroundingDINO",
                Path.cwd().parent / "GroundingDINO",
                Path.cwd() / "GroundingDINO",
            ]
        )
        for candidate in candidates:
            script = candidate / "fence_grounded_sam_1.py"
            if script.exists():
                if str(candidate.resolve()) not in sys.path:
                    sys.path.insert(0, str(candidate.resolve()))
                module_name = "fence_grounded_sam_1"
                loaded = sys.modules.get(module_name)
                if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == script.resolve():
                    return loaded
                spec = importlib.util.spec_from_file_location(module_name, script)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot load module spec from {script}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module
        raise FileNotFoundError("cannot locate GroundingDINO/fence_grounded_sam_1.py")

    @staticmethod
    def _resolve_external_path(path: str | Path, script_dir: Path) -> Path:
        """Resolve external model paths, preferring the GroundingDINO/SAM2 layout."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj.resolve()
        candidates = [
            script_dir / path_obj,
            script_dir.parent / path_obj,
            Path.cwd() / path_obj,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    def _resolve_sam2_config(self, fence: Any, sam2_repo: Path) -> Path:
        """Resolve SAM2 image-predictor config without picking STEAD wrapper files."""
        requested = self.sam2_config or "sam2/configs/sam2/sam2_hiera_l.yaml"
        requested_path = Path(requested)
        if requested_path.is_absolute() and requested_path.exists() and requested_path.stat().st_size > 256:
            return requested_path.resolve()
        name = requested_path.name
        candidates = [
            sam2_repo / "sam2" / "configs" / "sam2" / name,
            sam2_repo / "sam2" / requested,
            sam2_repo / requested,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.stat().st_size > 256:
                return candidate.resolve()
        return fence.resolve_sam2_config(sam2_repo, requested)

    def _mask_to_contours(self, mask: Any) -> list:
        try:
            from src.perception.sam_tracking_adapter import mask_to_contours

            return mask_to_contours(mask)
        except Exception:
            return []

    @staticmethod
    def _unavailable(reason: str, metadata: dict[str, Any] | None = None) -> GuardNetRegionResult:
        return GuardNetRegionResult(
            mask=None,
            contours=[],
            available=False,
            fallback_reason=reason,
            metadata={"enabled": True, "fallback_reason": reason, **(metadata or {})},
        )
