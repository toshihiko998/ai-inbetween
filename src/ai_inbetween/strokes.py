"""
strokes.py (robust)

Goals:
- Prefer centerline (skeleton) extraction to avoid outline-walking zigzags
- But NEVER return zero strokes silently:
    - auto-invert if binary looks inverted
    - relax thresholds
    - fallback to contour-based extraction when skeleton fails
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class Stroke:
    points: np.ndarray  # (N,2) float32

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.points, axis=0)

    @property
    def length(self) -> float:
        diffs = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))


# ---------- helpers ----------

def _remove_near_duplicates(points: np.ndarray, min_step: float = 0.75) -> np.ndarray:
    if len(points) <= 1:
        return points.astype(np.float32)
    out = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - out[-1]) >= min_step:
            out.append(p)
    return np.asarray(out, dtype=np.float32)


def _smooth_polyline(points: np.ndarray, window: int = 7) -> np.ndarray:
    pts = points.astype(np.float32)
    n = len(pts)
    if n < 3 or window <= 1:
        return pts
    if window % 2 == 0:
        window += 1
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return pts

    pad = w // 2
    kernel = np.ones(w, dtype=np.float32) / w

    x = np.pad(pts[:, 0], (pad, pad), mode="edge")
    y = np.pad(pts[:, 1], (pad, pad), mode="edge")
    xs = np.convolve(x, kernel, mode="valid")
    ys = np.convolve(y, kernel, mode="valid")

    sm = np.stack([xs, ys], axis=1).astype(np.float32)
    sm[0] = pts[0]
    sm[-1] = pts[-1]
    return sm


def _simplify_rdp(points: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    pts = points.astype(np.float32)
    if len(pts) < 3:
        return pts
    cnt = pts.reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=False)
    return approx.reshape(-1, 2).astype(np.float32)


def skeletonize(binary: np.ndarray, max_iters: int = 10000) -> np.ndarray:
    """
    Morphological skeletonization (no ximgproc).
    Input: uint8, lines=255 bg=0
    Output: uint8, skeleton lines=255 bg=0
    """
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    it = 0
    while True:
        it += 1
        if it > max_iters:
            break

        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded

        if cv2.countNonZero(img) == 0:
            break

    return skel


def _extract_by_contours(binary: np.ndarray, min_points: int, min_length: float) -> List[Stroke]:
    """Fallback: extract strokes directly from contours of the binary image."""
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    strokes: List[Stroke] = []
    for cnt in contours:
        if len(cnt) < min_points:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)
        pts = _remove_near_duplicates(pts, min_step=0.75)
        if len(pts) < min_points:
            continue
        stroke = Stroke(points=pts)
        if stroke.length < min_length:
            continue
        strokes.append(stroke)
    return strokes


# ---------- public API ----------

def extract_strokes(
    binary: np.ndarray,
    min_points: int = 8,
    min_length: float = 10.0,
    smooth_window: int = 7,
    simplify_epsilon: float = 1.0,
) -> List[Stroke]:
    """
    Robust stroke extraction:
    - auto-invert if needed
    - skeletonize
    - if skeleton has too few pixels, fallback to contour method
    """
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)

    h, w = binary.shape[:2]
    white = int(cv2.countNonZero(binary))
    total = int(h * w)

    # Auto-invert when binary seems inverted (background white, lines black)
    # If "white" occupies most of the image, likely inverted.
    if white > total * 0.6:
        binary = cv2.bitwise_not(binary)

    # ---- Connect broken lines (VERY IMPORTANT) ----
    # 1) close tiny gaps
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    prepared = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)

    # 2) bridge 1-2px breaks (light dilation then erode)
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    prepared = cv2.dilate(prepared, k_bridge, iterations=1)
    prepared = cv2.erode(prepared, k_bridge, iterations=1)

    # 3) remove isolated noise while keeping lines
    prepared = cv2.medianBlur(prepared, 3)

    # optional: thicken slightly before skeletonization
    prepared = cv2.dilate(prepared, k_close, iterations=1)

    skel = skeletonize(prepared)

    # If skeleton is almost empty, fallback
    skel_pixels = int(cv2.countNonZero(skel))
    if skel_pixels < 20:
        return _extract_by_contours(prepared, min_points=min_points, min_length=min_length)

    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    strokes: List[Stroke] = []
    for cnt in contours:
        if len(cnt) < min_points:
            continue

        pts = cnt.reshape(-1, 2).astype(np.float32)
        pts = _remove_near_duplicates(pts, min_step=0.75)
        if len(pts) < min_points:
            continue

        pts = _smooth_polyline(pts, window=smooth_window)
        pts = _simplify_rdp(pts, epsilon=simplify_epsilon)
        if len(pts) < min_points:
            continue

        stroke = Stroke(points=pts)
        if stroke.length < min_length:
            continue

        strokes.append(stroke)

    # Final fallback if still empty
    if not strokes:
        strokes = _extract_by_contours(prepared, min_points=min_points, min_length=min_length)

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
