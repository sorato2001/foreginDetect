"""ROI geometry helpers used by the STEAD rule engine."""

from __future__ import annotations

from math import hypot
from typing import Iterable


def point_in_polygon(point: Iterable[float], polygon: list[list[float]]) -> bool:
    """Return True when a point lies inside a polygon using ray casting."""
    points = list(polygon or [])
    if len(points) < 3:
        return False
    x, y = [float(v) for v in point]
    inside = False
    j = len(points) - 1
    for i, pi in enumerate(points):
        xi, yi = float(pi[0]), float(pi[1])
        xj, yj = float(points[j][0]), float(points[j][1])
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def signed_line_side(point: Iterable[float], a: Iterable[float], b: Iterable[float]) -> float:
    """Return the signed side of a point relative to directed line a->b."""
    px, py = [float(v) for v in point]
    ax, ay = [float(v) for v in a]
    bx, by = [float(v) for v in b]
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def distance(p1: Iterable[float], p2: Iterable[float]) -> float:
    """Return Euclidean distance between two points."""
    x1, y1 = [float(v) for v in p1]
    x2, y2 = [float(v) for v in p2]
    return hypot(x2 - x1, y2 - y1)

