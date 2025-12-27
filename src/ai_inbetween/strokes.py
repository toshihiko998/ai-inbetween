"""
strokes.py (FINAL: NO CIRCULAR IMPORT)

IMPORTANT:
- NEVER import ".strokes" in this file.
- Exports:
    Stroke
    extract_strokes(...)
    resample_polyline(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Stroke:
    points: np.ndarray  # (N,2) float32

    @property
    def centroid(self) -> np.ndarray:
        if self.points is None or len(self.points) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        return np.mean(self.points, axis=0).astype(np.float32)

    @property
    def length(self) -> float:
        if self.points is None or len(self.points) < 2:
            return 0.0
        d = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(d, axis=1)))

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        if self.points is None or len(self.points) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        mn = np.min(self.points, axis=0)
        mx = np.max(self.points, axis=0)
        return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1]))


def _remove_near_duplicates(points: np.ndarray, min_step: float = 0.5) -> np.ndarray:
    if len(points) <= 1:
        return points.astype(np.float32)
    out = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - out[-1]) >= min_step:
            out.append(p)
    return np.asarray(out, dtype=np.float32)


def _simplify_rdp(points: np.ndarray, epsilon: float = 0.6) -> np.ndarray:
    pts = points.astype(np.float32)
    if len(pts) < 3:
        return pts
    cnt = pts.reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=False)
    return approx.reshape(-1, 2).astype(np.float32)


def extract_strokes(
    binary: np.ndarray,
    min_points: int = 8,
    min_length: float = 6.0,
    close_ksize: int = 3,
    close_iter: int = 1,
    simplify_epsilon: float = 0.6,
) -> List[Stroke]:
    """
    binary: uint8, lines=255 bg=0 (auto invert if needed)
    """
    if binary is None:
        return []
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)

    h, w = binary.shape[:2]
    white = int(cv2.countNonZero(binary))
    total = int(h * w)

    # auto invert if background seems white-heavy
    if white > total * 0.6:
        binary = cv2.bitwise_not(binary)

    # mild close to connect gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    prepared = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=close_iter)

    contours, _ = cv2.findContours(prepared, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    strokes: List[Stroke] = []
    for cnt in contours:
        if len(cnt) < min_points:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)
        pts = _remove_near_duplicates(pts, 0.5)
        if len(pts) < min_points:
            continue
        pts = _simplify_rdp(pts, epsilon=simplify_epsilon)
        if len(pts) < min_points:
            continue
        s = Stroke(points=pts)
        if s.length < min_length:
            continue
        strokes.append(s)

    return strokes


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
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
