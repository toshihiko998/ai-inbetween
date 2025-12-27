"""
strokes.py

Stroke extraction utilities.

This module converts binary line-art images into
lists of strokes (polylines).

Rules:
- NO drawing
- NO interpolation
- Geometry extraction only
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class Stroke:
    """
    Represents a single stroke as a polyline.
    """
    points: np.ndarray  # shape: (N, 2), dtype: float32

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.points, axis=0)

    @property
    def length(self) -> float:
        diffs = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))


def extract_strokes(binary: np.ndarray, min_points: int = 10) -> List[Stroke]:
    """
    Extract strokes from a binary image.

    Input:
      binary: uint8 image
        - lines = 255
        - background = 0

    Output:
      List of Stroke objects
    """
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    strokes: List[Stroke] = []

    for cnt in contours:
        if len(cnt) < min_points:
            continue

        points = cnt.reshape(-1, 2).astype(np.float32)

        cleaned = [points[0]]
        for p in points[1:]:
            if np.linalg.norm(p - cleaned[-1]) >= 0.5:
                cleaned.append(p)

        if len(cleaned) < min_points:
            continue

        strokes.append(Stroke(points=np.array(cleaned, dtype=np.float32)))

    return strokes


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """
    Resample polyline to n points by arc length.
    """
    pts = np.asarray(points, dtype=np.float32)

    if len(pts) == 0:
        return np.zeros((n, 2), dtype=np.float32)

    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(dist[-1])

    if total < 1e-6:
        return np.repeat(pts[:1], n, axis=0)

    target = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float32)

    j = 0
    for i, t in enumerate(target):
        while j < len(dist) - 2 and dist[j + 1] < t:
            j += 1

        t0, t1 = dist[j], dist[j + 1]
        p0, p1 = pts[j], pts[j + 1]

        if abs(t1 - t0) < 1e-6:
            out[i] = p0
        else:
            a = (t - t0) / (t1 - t0)
            out[i] = (1.0 - a) * p0 + a * p1

    return out
