"""Simple IOU tracker fallback for offline experiments."""

from __future__ import annotations

from src.perception.detector import Detection


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


class SimpleIOUTracker:
    """Minimal class-aware IOU tracker used when ByteTrack is unavailable."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self.tracks: dict[int, list[Detection]] = {}
        self.next_id = 1

    def update(self, detections: list[Detection]) -> dict[int, list[Detection]]:
        """Assign detections to existing tracks by IOU."""
        for det in detections:
            best_id: int | None = None
            best_iou = 0.0
            for track_id, history in self.tracks.items():
                last = history[-1]
                if last.label != det.label:
                    continue
                score = _iou(last.bbox, det.bbox)
                if score > best_iou:
                    best_iou = score
                    best_id = track_id
            if best_id is not None and best_iou >= self.iou_threshold:
                self.tracks[best_id].append(det)
            else:
                self.tracks[self.next_id] = [det]
                self.next_id += 1
        return self.tracks

