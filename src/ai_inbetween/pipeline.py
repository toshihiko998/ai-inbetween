"""
pipeline.py (minimal safe version)

Purpose:
- Ensure in-between lines are ALWAYS rendered
- Avoid indentation errors
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .preprocess import load_grayscale, binarize_lineart
from .strokes import extract_strokes, resample_polyline
from .match import match_strokes
from .render import render_polylines
from .diagnostics import to_line_binary, overlap_metrics, jitter_score


def run_pipeline(
    image_a: Path,
    image_b: Path,
    out_dir: Path,
    inbetween_count: int = 5,
    thickness: int = 2,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_a = load_grayscale(image_a)
    img_b = load_grayscale(image_b)

    if img_a.shape != img_b.shape:
        raise ValueError("Keyframes must have same resolution")

    bin_a = binarize_lineart(img_a)
    bin_b = binarize_lineart(img_b)

    strokes_a = extract_strokes(bin_a)
    strokes_b = extract_strokes(bin_b)

    matches = match_strokes(strokes_a, strokes_b)

    # confidenceが一番高い1本だけ使う
    matches = sorted(matches, key=lambda m: m.confidence, reverse=True)[:1]


    prev_bin: Optional[np.ndarray] = None

    for i in range(1, inbetween_count + 1):
        alpha = i / (inbetween_count + 1)

        polylines: List[np.ndarray] = []

        for m in matches:
            sa = strokes_a[m.a_index].points
            sb = strokes_b[m.b_index].points

            pa = resample_polyline(sa, 64)
            pb = resample_polyline(sb, 64)

            interp = (1.0 - alpha) * pa + alpha * pb
            polylines.append(interp)

        frame = render_polylines(polylines, img_a.shape, thickness=thickness)

        inb_bin = to_line_binary(frame)
        om = overlap_metrics(inb_bin, bin_a, bin_b)
        jit = jitter_score(prev_bin, inb_bin)

        out_path = out_dir / f"{i:04d}.png"
        cv2.imwrite(str(out_path), frame)

        print(
            f"[frame {i:04d}] polylines={len(polylines)} "
            f"overlap={om['overlap_score']:.3f} jitter={jit:.2f}"
        )

        prev_bin = inb_bin
