"""
pipeline.py (COMPATIBLE DROP-IN)

- Must accept keyword args used by cli.py:
    run_pipeline(image_a=..., image_b=..., out_dir=..., inbetween_count=..., thickness=...)

- Uses:
    load_grayscale (preprocess.py)
    extract_strokes / resample_polyline (strokes.py)
    match_strokes (match.py)

- Output: out_dir/0001.png .. 000N.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np

from .preprocess import load_grayscale
from .strokes import extract_strokes, resample_polyline
from .match import match_strokes


def _binarize(gray: np.ndarray) -> np.ndarray:
    """
    Output: uint8 lines=255 bg=0
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    g = gray.astype(np.uint8)

    # adaptive threshold tends to keep thin lines
    bw = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    return bw


def _render(polylines: List[Tuple[np.ndarray, int]], shape_hw) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    canvas = np.full((h, w), 255, dtype=np.uint8)
    for pts, t in polylines:
        if pts is None or len(pts) < 2:
            continue
        p = np.asarray(pts, dtype=np.float32)
        p = np.round(p).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [p], False, 0, thickness=max(1, int(t)), lineType=cv2.LINE_AA)
    return canvas


def run_pipeline(
    image_a: Any,
    image_b: Any,
    out_dir: Any,
    inbetween_count: int = 5,
    thickness: int = 1,
    **kwargs: Any,
) -> None:
    """
    cli.py compatible entry point.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_a = load_grayscale(Path(image_a))
    img_b = load_grayscale(Path(image_b))

    if img_a is None or img_b is None:
        raise FileNotFoundError(f"Image not found: {image_a} or {image_b}")

    if img_a.shape != img_b.shape:
        raise ValueError(f"Resolution mismatch: A={img_a.shape}, B={img_b.shape}")

    # knobs
    n_points = int(kwargs.get("n_points", 64))
    main_k = int(kwargs.get("main_k", 1))
    sub_k = int(kwargs.get("sub_k", 3))

    bin_a = _binarize(img_a)
    bin_b = _binarize(img_b)

    strokes_a = extract_strokes(bin_a)
    strokes_b = extract_strokes(bin_b)

    matches = match_strokes(strokes_a, strokes_b)
    if not matches:
        print("No matches found.")
        return

    matches = sorted(matches, key=lambda m: getattr(m, "confidence", 0.0), reverse=True)
    main_matches = matches[: max(0, main_k)]
    sub_matches = matches[main_k : main_k + max(0, sub_k)]

    for i in range(1, int(inbetween_count) + 1):
        alpha = i / (int(inbetween_count) + 1)

        polylines: List[Tuple[np.ndarray, int]] = []

        # main line (1px)
        for m in main_matches:
            sa = strokes_a[m.a_index].points
            sb = strokes_b[m.b_index].points
            pa = resample_polyline(sa, n_points)
            pb = resample_polyline(sb, n_points)
            interp = (1.0 - alpha) * pa + alpha * pb
            polylines.append((interp, 1))

        # sub lines (1px, reduce count to make them "lighter")
        for m in sub_matches:
            sa = strokes_a[m.a_index].points
            sb = strokes_b[m.b_index].points
            pa = resample_polyline(sa, n_points)
            pb = resample_polyline(sb, n_points)
            interp = (1.0 - alpha) * pa + alpha * pb
            polylines.append((interp, 1))

        frame = _render(polylines, img_a.shape[:2])
        out_path = out_dir / f"{i:04d}.png"
        cv2.imwrite(str(out_path), frame)

        print(f"[frame {i:04d}] polylines={len(polylines)}")
