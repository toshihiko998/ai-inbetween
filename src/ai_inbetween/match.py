"""
match.py

Stroke correspondence estimation between keyframes.

Rules:
- NO drawing
- NO blending / crossfade / morph
- Output only "which stroke matches which"

This is intentionally simple (baseline) and designed to be replaced later.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .strokes import Stroke


@dataclass
class StrokeMatch:
    """
    Represents a correspondence between a stroke in A and a stroke in B.
    """
    a_index: int
    b_index: int
    confidence: float
    centroid_distance: float
    length_ratio: float


def _length_ratio(a_len: float, b_len: float) -> float:
    # 1.0 is perfect; bigger is more mismatch
    denom = max(a_len, b_len, 1e-6)
    return float(abs(a_len - b_len) / denom)


def match_strokes(
    strokes_a: List[Stroke],
    strokes_b: List[Stroke],
    length_weight: float = 50.0
) -> List[StrokeMatch]:
    """
    Greedy matching baseline:
    For each stroke in A, find the best unused stroke in B by:
      cost = centroid_distance + length_weight * length_ratio

    Returns:
      List of StrokeMatch (may be shorter than strokes_a if B runs out)
    """
    if not strokes_a or not strokes_b:
        return []

    unused_b = set(range(len(strokes_b)))
    matches: List[StrokeMatch] = []

    for i, sa in enumerate(strokes_a):
        best_j: Optional[int] = None
        best_cost = float("inf")
        best_cd = float("inf")
        best_lr = float("inf")

        ca = sa.centroid
        la = sa.length

        for j in list(unused_b):
            sb = strokes_b[j]
            cb = sb.centroid
            lb = sb.length

            cd = float(np.linalg.norm(ca - cb))
            lr = _length_ratio(la, lb)
            cost = cd + length_weight * lr

            if cost < best_cost:
                best_cost = cost
                best_j = j
                best_cd = cd
                best_lr = lr

        if best_j is None:
            continue

        unused_b.remove(best_j)

        # Map cost to a rough confidence in [0,1]
        # Smaller cost => higher confidence
        confidence = float(1.0 / (1.0 + best_cost / 50.0))

        matches.append(
            StrokeMatch(
                a_index=i,
                b_index=best_j,
                confidence=confidence,
                centroid_distance=best_cd,
                length_ratio=best_lr,
            )
        )

    return matches
