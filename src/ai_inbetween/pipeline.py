"""
pipeline.py

Main in-between pipeline (baseline).

Flow:
- load images
- preprocess (binarize)
- extract strokes
- match strokes
- generate intermediate strokes
- render frames
- compute diagnostics
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np

from .preprocess import load_grayscale, binarize_lineart
from .strokes import extract_strokes
from .match import match_strokes
from .render import render_polylines
from .diagnostics import (
    to_line_binary,
    overlap_metrics,
    jitter_score,
)


def run_pipeline(
    image_a: Path,
    image_b: Path,
    out_dir: Path,
    inbetween_count: int = 5,
    thickness: int = 2,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    img_a = load_grayscale(image_a)
    img_b = load_grayscale(image_b)

    # Preprocess
    bin_a = binarize_lineart(img_a)
    bin_b = binarize_lineart(img_b)

    # Extract strokes
    strokes_a = extract_strokes(bin_a)
    strokes_b = extract_strokes(bin_b)

    # Match strokes
    matches = match_strokes(strokes_a, strokes_b)

    prev_bin = None

    for i in range(1, inbetween_count + 1):
        alpha = i / (inbetween_count + 1)

        polylines: List[np.ndarray] = []

        for m in matches:
            sa = strokes_a[m.a_index].points
            sb = strokes_b[m.b_index].points

            n = min(len(sa), len(sb))
            if n < 2:
                continue

            interp = (1.0 - alpha) * sa[:n] + alpha * sb[:n]
            polylines.append(interp)

        # Render
        frame = render_polylines(polylines, img_a.shape, thickness=thickness)

        # Diagnostics
        inb_bin = to_line_binary(frame)
        metrics = overlap_metrics(inb_bin, bin_a, bin_b)
        jitter = jitter_score(prev_bin, inb_bin)

        # Save
        out_path = out_dir / f"{i:04d}.png"
        cv2.imwrite(str(out_path), frame)

        print(
            f"[frame {i}] overlap={metrics['overlap_score']:.3f} "
            f"jitter={jitter:.2f}"
        )

        prev_bin = inb_bin
