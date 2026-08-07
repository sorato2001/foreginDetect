import json
from pathlib import Path

import numpy as np

from src.perception.guard_net_rule_region import (
    GuardNetConfig,
    GuardNetRuleRegionGenerator,
    build_continuous_band,
    build_fitted_line_band,
    largest_mask_components,
)
from src.perception.open_vocab_detector import OpenVocabularyDetection
from src.perception.sam_tracking_adapter import SAMFrameResult, SAMTrackingAdapter, SAMTrackingConfig, SAMTrackingResult
from src.pipeline.analyze_event import (
    _resolve_rule_region_source,
    _sam_tracking_evidence_metadata,
)


def test_continuous_band_horizontal_mask():
    mask = np.zeros((40, 100), dtype=np.uint8)
    mask[10:25, 5:35] = 1
    mask[10:25, 65:95] = 1

    continuous, geometry = build_continuous_band(mask)

    assert geometry is not None
    assert continuous[15, 50] == 1
    assert continuous[:, geometry["x_min"] : geometry["x_max"] + 1].sum() > mask.sum()


def test_continuous_band_sloped_mask():
    mask = np.zeros((80, 120), dtype=np.uint8)
    for x in range(10, 111):
        top = 12 + x // 10
        bottom = top + 20
        mask[top : bottom + 1, x] = 1

    continuous, geometry = build_continuous_band(mask, side_fraction=0.1)

    assert continuous is not None
    assert geometry["left_top"][0] == 10
    assert geometry["right_top"][0] == 110
    assert geometry["left_bottom"][1] > geometry["left_top"][1]
    assert geometry["right_bottom"][1] > geometry["right_top"][1]
    assert geometry["top_line"]["slope"] > 0
    assert geometry["bottom_line"]["slope"] > 0
    assert len(geometry["polygon"]) == 4


def test_continuous_band_ignores_internal_occlusion():
    mask = np.zeros((60, 100), dtype=np.uint8)
    mask[15:45, 10:90] = 1
    mask[20:40, 40:60] = 0

    continuous, _ = build_continuous_band(mask)

    assert continuous is not None
    assert continuous[30, 50] == 1


def test_continuous_band_does_not_require_connected_mask():
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[10:30, 5:20] = 1
    mask[12:32, 45:55] = 1
    mask[14:34, 80:95] = 1

    continuous, _ = build_continuous_band(mask)

    assert continuous is not None
    assert continuous[22, 35] == 1
    assert continuous[24, 70] == 1


def test_empty_mask_fails_cleanly():
    continuous, geometry = build_continuous_band(np.zeros((20, 20), dtype=np.uint8))

    assert continuous is None
    assert geometry is None


def test_mask_all_keeps_largest_component_and_fits_boundary_lines():
    mask_all = np.zeros((80, 120), dtype=np.uint8)
    for x in range(10, 111):
        top = 18 + x // 12
        mask_all[top : top + 24, x] = 1
    mask_all[3:7, 3:7] = 1

    refined, components = largest_mask_components(mask_all)
    band, geometry = build_fitted_line_band(refined)

    assert len(components) == 1
    assert not refined[4, 4]
    assert band is not None
    assert geometry["mode"] == "mask_all_largest_component_ransac_lines"
    assert geometry["top_line"]["inliers"] > 50
    assert geometry["bottom_line"]["inliers"] > 50
    assert geometry["top_line"]["slope"] > 0


class _Detector:
    def detect(self, frame, text_prompt):
        return [
            OpenVocabularyDetection([5, 12, 55, 42], 0.8, "fence"),
            OpenVocabularyDetection([45, 12, 95, 42], 0.75, "fence"),
        ]


class _Predictor:
    def set_image(self, image):
        self.shape = image.shape[:2]

    def predict(self, box, multimask_output=True):
        height, width = self.shape
        mask = np.zeros((height, width), dtype=bool)
        if float(np.asarray(box).reshape(-1)[0]) < 20:
            mask[18:38, 8:56] = True
            mask[2:5, 2:5] = True
        else:
            mask[18:38, 45:92] = True
        return np.asarray([mask]), np.asarray([0.9]), None


def _generator(output_dir=None, export=False):
    return GuardNetRuleRegionGenerator(
        GuardNetConfig(output_dir=str(output_dir) if output_dir else None, export_yolo_seg=export),
        device="cpu",
        detector=_Detector(),
        predictor_factory=_Predictor,
    )


def test_all_candidate_box_masks_are_merged_before_refine():
    result = _generator().generate(np.zeros((60, 100, 3), dtype=np.uint8))

    assert result.available
    assert result.selected_candidate is None
    assert len(result.candidate_scores) == 2
    assert result.mask_all[3, 3]
    assert not result.refined_mask[3, 3]
    assert result.refined_mask[25, 20]
    assert result.refined_mask[25, 80]
    assert result.metadata["aggregation_mode"] == "mask_all_largest_component_ransac_lines"


def test_rule_region_source_guard_net_and_auto_priority():
    assert _resolve_rule_region_source("guard_net", "sam_tracking", None) == "guard_net"
    assert _resolve_rule_region_source("auto", "sam_tracking", "track.pt", True) == "guard_net"
    assert _resolve_rule_region_source("auto", "sam_tracking", "track.pt", False) == "sam_track"
    assert _resolve_rule_region_source("auto", "sam_tracking", None, False) == "yaml"
    assert _resolve_rule_region_source("auto", "simple_iou", "track.pt", True) == "yaml"


def test_guard_net_filters_oversized_detection_box():
    generator = _generator()
    detections = [
        OpenVocabularyDetection([0, 0, 100, 60], 0.9, "oversized fence"),
        OpenVocabularyDetection([10, 10, 60, 40], 0.7, "local fence"),
    ]

    filtered = generator._filter_detections(detections, width=100, height=60)

    assert len(filtered) == 1
    assert filtered[0].phrase == "local fence"


def _fallback_adapter(with_track_model=True, requested="auto"):
    adapter = SAMTrackingAdapter.__new__(SAMTrackingAdapter)
    adapter.config = SAMTrackingConfig(
        rule_region_source="guard_net",
        rule_region_source_requested=requested,
        sam_track_fallback_enabled=requested == "auto",
    )
    adapter._resolved_rule_region_source = "guard_net"
    adapter._track_model = object() if with_track_model else None
    adapter._guard_net_error = "groundingdino_unavailable"
    adapter._detect_guard_net_mask = lambda frame: None
    return adapter


def test_auto_falls_back_to_sam_track():
    adapter = _fallback_adapter(with_track_model=True)
    expected = np.ones((4, 4), dtype=np.uint8)
    adapter._detect_sam_track_mask = lambda frame: expected

    actual = adapter.detect_track_mask(np.zeros((4, 4, 3), dtype=np.uint8))

    assert actual is expected
    assert adapter._resolved_rule_region_source == "sam_track"


def test_auto_falls_back_to_yaml():
    adapter = _fallback_adapter(with_track_model=False)

    assert adapter.detect_track_mask(np.zeros((4, 4, 3), dtype=np.uint8)) is None
    assert adapter._resolved_rule_region_source == "yaml"


def test_explicit_guard_net_failure_allows_yaml_fallback():
    adapter = _fallback_adapter(with_track_model=True, requested="guard_net")

    assert adapter.detect_track_mask(np.zeros((4, 4, 3), dtype=np.uint8)) is None
    assert adapter._resolved_rule_region_source == "yaml"


def test_guard_net_yolo_seg_normalized(tmp_path):
    result = _generator(tmp_path, export=True).generate(np.zeros((60, 100, 3), dtype=np.uint8))

    assert result.available
    values = (tmp_path / "continuous_band_yolo_seg.txt").read_text(encoding="utf-8").split()
    assert values[0] == "0"
    assert all(0.0 <= float(value) <= 1.0 for value in values[1:])
    saved = json.loads((tmp_path / "guard_net_result.json").read_text(encoding="utf-8"))
    assert saved["metadata"]["continuous_band"]["mode"] == "mask_all_largest_component_ransac_lines"


def test_guard_net_evidence_visible_to_vlm():
    result = SAMTrackingResult(
        tracks={},
        fps=25.0,
        duration=1.0,
        frames=[
            SAMFrameResult(
                frame_index=0,
                timestamp=0.0,
                detections=[],
                object_mask_count=1,
                track_mask_available=True,
                max_iou=0.3,
                max_object_overlap=0.6,
                suspicious=True,
                window_count=1,
                alarm=True,
            )
        ],
        metadata={
            "rule_region_source_resolved": "guard_net",
            "rule_region_available": True,
            "track_mask_seen": True,
            "guard_net": {
                "text_prompt": "fence",
                "source_frame_index": 5,
                "aggregate_confidence": 0.88,
                "continuous_band": {"polygon": [[1, 2], [9, 2], [9, 8], [1, 8]], "continuous_mask_area": 48},
            },
        },
    )

    metadata = _sam_tracking_evidence_metadata(result, "sam_tracking_result.json")

    assert metadata["rule_region"]["source"] == "guard_net"
    assert metadata["rule_region"]["confidence"] == 0.88
    assert metadata["rule_region"]["continuous_band_area"] == 48
    assert metadata["person_mask_intrusion_evidence"]["max_object_overlap"] == 0.6
