"""
pipeline.py (safe baseline)

- No blending/crossfade
- Extract strokes -> match -> resample -> interpolate -> render
- Confidence gating is temporarily DISABLED to ensure output lines appear.
"""

from __future__ import annotations

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
    n_points: int = 64,
    # match weights (tune later)
    w_centroid: float = 1.0,
    w_len: float = 15.0,
    w_angle: float = 5.0,
    w_shape: float = 8.0,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_a = load_grayscale(image_a)
    img_b = load_grayscale(image_b)

    if img_a.shape != img_b.shape:
        raise ValueError(f"Resolution mismatch: A={img_a.shape}, B={img_b.shape}")

    bin_a = binarize_lineart(img_a)  # lines=255, bg=0
    bin_b = binarize_lineart(img_b)

    strokes_a = extract_strokes(bin_a)
    strokes_b = extract_strokes(bin_b)

    matches = match_strokes(
        strokes_a,
        strokes_b,
        w_centroid=w_centroid,
        w_len=w_len,
        w_angle=w_angle,
        w_shape=w_shape,
        shape_points=n_points,
    )

    prev_bin: Optional[np.ndarray] = None

    for i in range(1, inbetween_count + 1):
        alpha = i / (inbetween_count + 1)

        polylines: List[np.ndarray] = []

        for m in matches:
            # confidence gate is disabled for now:
            # if m.confidence < 0.35:
            #     continue

            sa = strokes_a[m.a_index].points
            sb = strokes_b[m.b_index].points

            pa = resample_polyline(sa, n_points)
            pb = resample_polyline(sb, n_points)

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
