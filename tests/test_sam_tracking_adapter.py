from pathlib import Path

import numpy as np

from src.perception.sam_tracking_adapter import SlidingWindowIntrusionJudge, _resolve_sam2_config, bbox_to_mask


def test_mask_iou_and_bbox_mask():
    mask_a = bbox_to_mask((20, 20), [0, 0, 10, 10])
    mask_b = bbox_to_mask((20, 20), [5, 5, 15, 15])

    score = SlidingWindowIntrusionJudge.compute_iou(mask_a, mask_b)

    assert np.isclose(score, 25 / 175)


def test_sliding_window_confirms_intrusion():
    judge = SlidingWindowIntrusionJudge(iou_threshold=0.10, window_size=5, confirm_count=3)

    states = [judge.update(idx, score) for idx, score in enumerate([0.0, 0.2, 0.15, 0.0, 0.3])]

    assert states[-1]["alarm"] is True
    events = judge.get_events()
    assert len(events) == 1
    assert events[0].peak_iou == 0.3


def test_object_overlap_detects_small_object_inside_large_track():
    judge = SlidingWindowIntrusionJudge(iou_threshold=0.10, object_overlap_threshold=0.15, window_size=3, confirm_count=1)

    state = judge.update(frame_index=1, max_iou=0.02, max_object_overlap=0.8)

    assert state["suspicious"] is True
    assert state["alarm"] is True


def test_resolve_sam2_config_skips_pointer_file():
    config_dir, config_name = _resolve_sam2_config("sam2_hiera_l.yaml")

    assert config_name == "sam2_hiera_l"
    config_path = Path(config_dir) / f"{config_name}.yaml"
    assert config_path.exists()
    assert "configs/sam2" not in config_path.read_text(encoding="utf-8").strip()


def test_track_mask_labels_filter_segmentation_masks():
    class _Array:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class _Masks:
        def __init__(self):
            self.data = [
                _Array(np.ones((8, 8), dtype=np.float32)),
                _Array(np.zeros((8, 8), dtype=np.float32)),
            ]

    class _Boxes:
        cls = _Array(np.array([0, 1], dtype=np.float32))

    class _Result:
        names = {0: "fence", 1: "track"}
        masks = _Masks()
        boxes = _Boxes()

    class _Model:
        names = {0: "fence", 1: "track"}

        def predict(self, **kwargs):
            return [_Result()]

    from src.perception.sam_tracking_adapter import SAMTrackingAdapter, SAMTrackingConfig

    adapter = SAMTrackingAdapter.__new__(SAMTrackingAdapter)
    adapter.config = SAMTrackingConfig(track_mask_labels=["track"])
    adapter._track_model = _Model()
    adapter._runtime_device = "cpu"

    mask = adapter.detect_track_mask(np.zeros((8, 8, 3), dtype=np.uint8))

    assert mask is None


def test_track_mask_labels_accept_numeric_class_ids():
    class _Array:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class _Masks:
        data = [_Array(np.ones((8, 8), dtype=np.float32))]

    class _Boxes:
        cls = _Array(np.array([1], dtype=np.float32))

    class _Result:
        names = {0: "fence", 1: "track"}
        masks = _Masks()
        boxes = _Boxes()

    class _Model:
        names = {0: "fence", 1: "track"}

        def predict(self, **kwargs):
            return [_Result()]

    from src.perception.sam_tracking_adapter import SAMTrackingAdapter, SAMTrackingConfig

    adapter = SAMTrackingAdapter.__new__(SAMTrackingAdapter)
    adapter.config = SAMTrackingConfig(track_mask_labels=["1"])
    adapter._track_model = _Model()
    adapter._runtime_device = "cpu"

    mask = adapter.detect_track_mask(np.zeros((8, 8, 3), dtype=np.uint8))

    assert mask is not None
    assert int(mask.sum()) == 64


def test_sam2_video_tracker_accepts_numpy_prompt_boxes(monkeypatch):
    from src.perception import sam_tracking_adapter
    from src.perception.sam_tracking_adapter import _SAM2VideoMemoryTracker

    class _Predictor:
        def __init__(self):
            self.boxes = []

        def init_state(self, **kwargs):
            return {}

        def add_new_points_or_box(self, **kwargs):
            self.boxes.append(kwargs["box"])

        def propagate_in_video(self, state):
            return iter(())

    predictor = _Predictor()
    monkeypatch.setattr(_SAM2VideoMemoryTracker, "_load_model", lambda self: predictor)
    monkeypatch.setattr(sam_tracking_adapter, "_box_iou", lambda box, registered: 0.0)
    tracker = _SAM2VideoMemoryTracker(
        sam2_config="sam2_hiera_l.yaml",
        sam2_checkpoint="weights/sam2_hiera_large.pt",
        device="cpu",
    )

    tracker.initialize_video(
        "examples/test1.mp4",
        scan_prompts=[{"boxes": np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.float32)}],
    )

    assert len(predictor.boxes) == 2
    assert tracker._has_objects is True


def test_sam2_video_tracker_advances_to_requested_frame(monkeypatch):
    from src.perception import sam_tracking_adapter
    from src.perception.sam_tracking_adapter import _SAM2VideoMemoryTracker

    class _Predictor:
        def init_state(self, **kwargs):
            return {}

        def add_new_points_or_box(self, **kwargs):
            return None

        def propagate_in_video(self, state):
            for frame_idx in range(5):
                yield frame_idx, [9], np.ones((1, 1, 4, 4), dtype=np.float32) * frame_idx

    monkeypatch.setattr(_SAM2VideoMemoryTracker, "_load_model", lambda self: _Predictor())
    monkeypatch.setattr(sam_tracking_adapter, "_box_iou", lambda box, registered: 0.0)
    tracker = _SAM2VideoMemoryTracker(
        sam2_config="sam2_hiera_l.yaml",
        sam2_checkpoint="weights/sam2_hiera_large.pt",
        device="cpu",
    )
    tracker.initialize_video("examples/test1.mp4", scan_prompts=[{"boxes": np.array([[1, 2, 3, 4]], dtype=np.float32)}])

    masks, _scores, ids = tracker.track_frame(3)

    assert ids == [9]
    assert tracker._last_output_frame_idx == 3
    assert len(masks) == 1
