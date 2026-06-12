import numpy as np

from src.perception.sam_tracking_adapter import SlidingWindowIntrusionJudge, bbox_to_mask


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
