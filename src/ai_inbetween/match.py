"""
match.py

Improved stroke correspondence estimation between keyframes.

Adds:
- orientation (PCA angle)
- shape distance (after resampling + normalization)
- confidence gating support (downstream)

Rules:
- NO drawing
- NO blending/crossfade/morph
- Output only matching pairs
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .strokes import Stroke, resample_polyline


@dataclass
class StrokeMatch:
    a_index: int
    b_index: int
    confidence: float
    cost: float
    centroid_distance: float
    length_ratio: float
    angle_diff: float
    shape_dist: float


def _length_ratio(a_len: float, b_len: float) -> float:
    denom = max(a_len, b_len, 1e-6)
    return float(abs(a_len - b_len) / denom)


def _pca_angle(points: np.ndarray) -> float:
    """
    Return principal direction angle in radians [-pi, pi].
    """
    pts = np.asarray(points, dtype=np.float32)
    c = pts.mean(axis=0, keepdims=True)
    x = pts - c
    # covariance
    cov = (x.T @ x) / max(len(x), 1)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    return float(np.arctan2(v[1], v[0]))


def _angle_diff(a: float, b: float) -> float:
    """
    Smallest angle difference in radians [0, pi].
    """
    d = abs(a - b)
    d = d % (2 * np.pi)
    if d > np.pi:
        d = 2 * np.pi - d
    return float(d)


def _normalize_shape(points: np.ndarray, n: int = 64) -> np.ndarray:
    """
    Resample -> center -> scale to unit RMS.
    """
    p = resample_polyline(points, n)
    p = p.astype(np.float32)
    p = p - p.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(p * p, axis=1))) + 1e-6)
    p = p / rms
    return p


def _shape_distance(pa: np.ndarray, pb: np.ndarray) -> float:
    """
    Simple shape distance: mean L2 between corresponding resampled points.
    (Not rotation-invariant; we use PCA angle separately.)
    """
    return float(np.mean(np.linalg.norm(pa - pb, axis=1)))


def match_strokes(
    strokes_a: List[Stroke],
    strokes_b: List[Stroke],
    w_centroid: float = 1.0,
    w_len: float = 40.0,
    w_angle: float = 15.0,
    w_shape: float = 25.0,
    shape_points: int = 64,
) -> List[StrokeMatch]:
    """
    Greedy matching with improved cost:

      cost =
        w_centroid * centroid_distance +
        w_len      * length_ratio +
        w_angle    * angle_diff +
        w_shape    * shape_dist

    Returns list of matches. Designed to be upgraded later to global assignment.
    """
    if not strokes_a or not strokes_b:
        return []

    # Precompute features for speed
    feats_a: List[Tuple[np.ndarray, float, float, np.ndarray]] = []
    for s in strokes_a:
        c = s.centroid.astype(np.float32)
        L = float(s.length)
        ang = _pca_angle(s.points)
        shp = _normalize_shape(s.points, n=shape_points)
        feats_a.append((c, L, ang, shp))

    feats_b: List[Tuple[np.ndarray, float, float, np.ndarray]] = []
    for s in strokes_b:
        c = s.centroid.astype(np.float32)
        L = float(s.length)
        ang = _pca_angle(s.points)
        shp = _normalize_shape(s.points, n=shape_points)
        feats_b.append((c, L, ang, shp))

    unused_b = set(range(len(strokes_b)))
    matches: List[StrokeMatch] = []

    for i, (ca, la, anga, sha) in enumerate(feats_a):
        best_j: Optional[int] = None
        best_cost = float("inf")
        best_cd = best_lr = best_ad = best_sd = float("inf")

        for j in list(unused_b):
            cb, lb, angb, shb = feats_b[j]

            cd = float(np.linalg.norm(ca - cb))
            lr = _length_ratio(la, lb)
            ad = _angle_diff(anga, angb)
            sd = _shape_distance(sha, shb)

            cost = (
                w_centroid * cd +
                w_len * lr +
                w_angle * ad +
                w_shape * sd
            )

            if cost < best_cost:
                best_cost = cost
                best_j = j
                best_cd, best_lr, best_ad, best_sd = cd, lr, ad, sd

        if best_j is None:
            continue

        unused_b.remove(best_j)

        # Confidence: inverse-ish mapping (tuned)
        confidence = float(1.0 / (1.0 + best_cost / 60.0))

        matches.append(
            StrokeMatch(
                a_index=i,
                b_index=best_j,
                confidence=confidence,
                cost=float(best_cost),
                centroid_distance=float(best_cd),
                length_ratio=float(best_lr),
                angle_diff=float(best_ad),
                shape_dist=float(best_sd),
            )
        )

    return matches
