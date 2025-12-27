"""
render.py

Render stroke polylines into a single clean line-art image.

Rules:
- White background (255)
- Black lines (0)
- Constant thickness
- No overlay of multiple frames (we only render one result per frame)
"""

from typing import Iterable, Tuple

import cv2
import numpy as np


def render_polylines(
    polylines: Iterable[np.ndarray],
    canvas_shape: Tuple[int, int],
    thickness: int = 2
) -> np.ndarray:
    """
    Render a list of polylines onto a white canvas.

    Args:
        polylines: each polyline is (N, 2) float/int array
        canvas_shape: (H, W)
        thickness: line thickness in pixels

    Returns:
        Grayscale uint8 image (H, W): white background, black lines
    """
    h, w = canvas_shape
    canvas = np.full((h, w), 255, dtype=np.uint8)

    for poly in polylines:
        if poly is None:
            continue
        poly = np.asarray(poly)
        if poly.ndim != 2 or poly.shape[0] < 2 or poly.shape[1] != 2:
            continue

        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)

        # Anti-aliased polyline rendering
        cv2.polylines(
            canvas,
            [pts],
            isClosed=False,
            color=0,
            thickness=int(thickness),
            lineType=cv2.LINE_AA
        )

    return canvas


def apply_lock_mask(
    base_img: np.ndarray,
    locked_from: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Replace pixels in base_img with locked_from where mask is "on".

    mask: uint8 or bool array. "on" means > 127 (or True).
    """
    if mask.dtype != np.bool_:
        m = mask > 127
    else:
        m = mask

    out = base_img.copy()
    out[m] = locked_from[m]
    return out
