"""Railway perimeter region-rule engine.

The rule engine combines object type, confidence, warning/danger polygons, and a
fence line to produce explainable risk levels. It deliberately avoids the naïve
"person detected equals alarm" logic; targets must have a meaningful spatial
relationship with the configured railway perimeter regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

RISK_ORDER = {"normal": 0, "low": 1, "medium": 2, "high": 3}


@dataclass(slots=True)
class RegionRuleResult:
    """Structured result from region-rule evaluation."""

    risk_level: str
    risk_score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "risk_score": round(float(self.risk_score), 4),
            "reason": self.reason,
        }


def point_in_polygon(point: Iterable[float], polygon: list[list[float]] | None) -> bool:
    """Return True if point is inside a polygon using ray casting."""
    if not polygon or len(polygon) < 3:
        return False
    x, y = [float(v) for v in point]
    inside = False
    j = len(polygon) - 1
    for i, pi in enumerate(polygon):
        xi, yi = float(pi[0]), float(pi[1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def distance_point_to_segment(point: Iterable[float], a: Iterable[float], b: Iterable[float]) -> float:
    """Compute Euclidean distance from a point to a line segment."""
    px, py = [float(v) for v in point]
    ax, ay = [float(v) for v in a]
    bx, by = [float(v) for v in b]
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


class RegionRules:
    """Evaluate railway perimeter rules for YOLO detections."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.image_size = config.get("image_size") or [1920, 1080]
        self.fence_line = config.get("fence_line") or []
        self.warning_zone = config.get("warning_zone") or []
        self.danger_zone = config.get("danger_zone") or []
        self.near_margin_px = float(config.get("near_margin_px", 80))

    def annotate_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add zone and fence-crossing information to detections."""
        enriched: list[dict[str, Any]] = []
        for det in detections:
            item = dict(det)
            center = item.get("center") or self._bbox_center(item.get("bbox") or [0, 0, 0, 0])
            item["center"] = [round(float(center[0]), 2), round(float(center[1]), 2)]
            if point_in_polygon(center, self.danger_zone):
                zone = "danger"
            elif point_in_polygon(center, self.warning_zone):
                zone = "warning"
            else:
                zone = "safe"
            item["zone"] = zone
            item["cross_fence"] = self.crosses_fence(item.get("bbox") or [0, 0, 0, 0], center)
            enriched.append(item)
        return enriched

    @staticmethod
    def _bbox_center(bbox: list[float]) -> list[float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    def crosses_fence(self, bbox: list[float], center: Iterable[float]) -> bool:
        """Heuristically decide whether a bbox crosses or is beyond the fence line."""
        if len(self.fence_line) < 2:
            return False
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx, cy = [float(v) for v in center]
        # For each fence segment, interpolate the line y at bbox center x.
        for idx in range(len(self.fence_line) - 1):
            ax, ay = [float(v) for v in self.fence_line[idx]]
            bx, by = [float(v) for v in self.fence_line[idx + 1]]
            if min(ax, bx) - 1 <= cx <= max(ax, bx) + 1:
                if ax == bx:
                    fence_y = (ay + by) / 2.0
                else:
                    ratio = (cx - ax) / (bx - ax)
                    fence_y = ay + ratio * (by - ay)
                # bbox touches the line or center is on the danger/track side (usually below the line).
                if y1 <= fence_y <= y2 or cy >= fence_y:
                    return True
        return False

    def _near_warning_or_fence(self, center: Iterable[float]) -> bool:
        """Return True if the point is near warning-zone boundary or fence line."""
        if self.warning_zone:
            for i in range(len(self.warning_zone)):
                if distance_point_to_segment(center, self.warning_zone[i], self.warning_zone[(i + 1) % len(self.warning_zone)]) <= self.near_margin_px:
                    return True
        if len(self.fence_line) >= 2:
            for i in range(len(self.fence_line) - 1):
                if distance_point_to_segment(center, self.fence_line[i], self.fence_line[i + 1]) <= self.near_margin_px:
                    return True
        return False

    def evaluate(self, detections: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate enriched detections and return the region-rule result."""
        if not detections:
            return RegionRuleResult("normal", 0.0, "未检测到有效目标，按区域规则判定为正常。 ").to_dict()

        max_score = 0.0
        max_level = "normal"
        reasons: list[str] = []
        target_cn = {"person": "人员", "animal": "动物", "car": "车辆", "truck": "车辆", "bus": "车辆", "motorcycle": "车辆", "bicycle": "车辆"}

        for det in detections:
            cls = str(det.get("class_name", "unknown"))
            conf = float(det.get("confidence", 0.0))
            zone = str(det.get("zone", "safe"))
            cross = bool(det.get("cross_fence", False))
            center = det.get("center") or [0, 0]
            level = "normal"
            score = 0.05

            if zone == "danger" or cross:
                level = "high" if cls == "person" else "medium"
                score = 0.85 if cls == "person" else 0.75
                reasons.append(f"{target_cn.get(cls, cls)}进入危险区或疑似跨越护网")
            elif zone == "warning":
                level = "medium"
                score = 0.55 if cls in {"person", "animal"} else 0.5
                reasons.append(f"{target_cn.get(cls, cls)}进入警戒区")
            elif self._near_warning_or_fence(center):
                level = "low"
                score = 0.3
                reasons.append(f"{target_cn.get(cls, cls)}靠近警戒区或护网")
            else:
                reasons.append(f"{target_cn.get(cls, cls)}位于安全区外，暂不构成入侵")

            # Confidence modulates score but does not erase high-risk geometry.
            score = min(1.0, score * (0.7 + min(conf, 1.0) * 0.3))
            if RISK_ORDER[level] > RISK_ORDER[max_level] or score > max_score:
                max_level = level
                max_score = score

        if max_level == "normal" and any(r for r in reasons):
            max_score = max(max_score, 0.1)
        return RegionRuleResult(max_level, max_score, "；".join(reasons[:3])).to_dict()
