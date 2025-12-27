"""
pipeline.py (stable)

- accepts inbetween_count keyword (required by cli.py)
- main line (top1 confidence) + sub lines (next 3) rendered thin
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

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
    thickness: int = 1,
    n_points: int = 64,
    main_k: int = 1,
    sub_k: int = 3,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_a = load_grayscale(image_a)
    img_b = load_grayscale(image_b)

    if img_a is None or img_b is None:
        raise FileNotFoundError(f"Image not found: {image_a} or {image_b}")

    if img_a.shape != img_b.shape:
        raise ValueError(f"Resolution mismatch: A={img_a.shape}, B={img_b.shape}")

    bin_a = binarize_lineart(img_a)  # lines=255, bg=0
    bin_b = binarize_lineart(img_b)

    strokes_a = extract_strokes(bin_a)
    strokes_b = extract_strokes(bin_b)

    matches = match_strokes(strokes_a, strokes_b)
    if not matches:
        print("No matches found.")
        return

    # 主線1本 + 補助線
    matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
    main_matches = matches[: max(0, int(main_k))]
    sub_matches = matches[int(main_k) : int(main_k) + max(0, int(sub_k))]

    prev_bin: Optional[np.ndarray] = None

    for i in range(1, inbetween_count + 1):
        alpha = i / (inbetween_count + 1)

        polylines: List[Tuple[np.ndarray, int]] = []

        # 主線（細い）
        for m in main_matches:
            sa = strokes_a[m.a_index].points
            sb = strokes_b[m.b_index].points
            pa = resample_polyline(sa, n_points)
            pb = resample_polyline(sb, n_points)
            interp = (1.0 - alpha) * pa + alpha * pb
            polylines.append((interp, 1))

        # 補助線（さらに細い＝同じ1でOK。控えめにするなら本数を減らすのが効く）
        for m in sub_matches:
            sa = strokes_a[m.a_index].points
            sb = strokes_b[m.b_index].points
            pa = resample_polyline(sa, n_points)
            pb = resample_polyline(sb, n_points)
            interp = (1.0 - alpha) * pa + alpha * pb
            polylines.append((interp, 1))

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
