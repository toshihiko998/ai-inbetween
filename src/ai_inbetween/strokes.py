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
    points: np.ndarray  # shape: (N, 2), dtype: int or float

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

    # Find contours (each contour ~= one stroke)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    strokes: List[Stroke] = []

    for cnt in contours:
        if len(cnt) < min_points:
            continue

        # cnt shape: (N, 1, 2) -> (N, 2)
        points = cnt.reshape(-1, 2).astype(np.float32)

        # Remove near-duplicate points
        cleaned = [points[0]]
        for p in points[1:]:
            if np.linalg.norm(p - cleaned[-1]) >= 0.5:
                cleaned.append(p)

        if len(cleaned) < min_points:
            continue

        strokes.append(Stroke(points=np.array(cleaned)))

    return strokes
