"""
diagnostics.py

Quality diagnostics for in-between results.

We measure:
- overlap/ghosting (looks like it contains both A and B)
- jitter (frame-to-frame instability)
- face drift (difference from A inside face lock mask)

These are heuristics. They are meant for logging and gating outputs,
not for perfect evaluation.
"""

from __future__ import annotations

from typing import Optional, Dict

import cv2
import numpy as np


def to_line_binary(gray_lineart: np.ndarray) -> np.ndarray:
    """
    Convert grayscale lineart (white bg, black lines) to binary lines=255, bg=0.
    """
    _, b = cv2.threshold(gray_lineart, 250, 255, cv2.THRESH_BINARY_INV)
    return b


def overlap_metrics(inb_bin: np.ndarray, a_bin: np.ndarray, b_bin: np.ndarray) -> Dict[str, float]:
    """
    Heuristic overlap/ghosting measure.

    Compute how much of in-between pixels overlap with A and B separately.
    If both ratios are high, output may look like "overlay" of A and B.

    Returns:
      intersectA_ratio, intersectB_ratio, overlap_score
    """
    inb_count = max(cv2.countNonZero(inb_bin), 1)
    ia = cv2.countNonZero(cv2.bitwise_and(inb_bin, a_bin)) / inb_count
    ib = cv2.countNonZero(cv2.bitwise_and(inb_bin, b_bin)) / inb_count
    overlap_score = float(min(ia, ib))  # both-high => higher ghosting suspicion
    return {
        "intersectA_ratio": float(ia),
        "intersectB_ratio": float(ib),
        "overlap_score": overlap_score,
    }


def jitter_score(prev_bin: Optional[np.ndarray], curr_bin: np.ndarray) -> float:
    """
    Simple jitter proxy using distance transforms.
    Lower is better. Roughly: average distance from stroke pixels to previous frame.
    """
    if prev_bin is None:
        return 0.0

    prev = (prev_bin > 0).astype(np.uint8)
    curr = (curr_bin > 0).astype(np.uint8)

    if prev.sum() == 0 or curr.sum() == 0:
        return 999.0

    dist_to_prev = cv2.distanceTransform(1 - prev, cv2.DIST_L2, 3)
    d1 = float(dist_to_prev[curr > 0].mean())

    dist_to_curr = cv2.distanceTransform(1 - curr, cv2.DIST_L2, 3)
    d2 = float(dist_to_curr[prev > 0].mean())

    return float((d1 + d2) * 0.5)


def face_drift_ratio(inb_bin: np.ndarray, a_bin: np.ndarray, face_mask: Optional[np.ndarray]) -> float:
    """
    If face_mask is provided, compare in-between vs A inside the mask.

    Returns a normalized ratio of differences (XOR) to A's stroke pixels.
    """
    if face_mask is None:
        return 0.0

    m = face_mask > 127
    if m.sum() == 0:
        return 0.0

    xor = cv2.bitwise_xor(inb_bin, a_bin)

    denom = max(int(np.count_nonzero(a_bin[m])) + 1, 1)
    return float(np.count_nonzero(xor[m]) / denom)
