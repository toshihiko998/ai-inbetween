"""
pipeline.py

Main in-between pipeline (baseline + improvements).

Flow:
- load images (A,B)
- preprocess (binarize)
- extract strokes
- match strokes
- generate in-between polylines (resample -> interpolate)
- render frames
- diagnostics per frame (overlap/jitter)
- save PNG sequence

Notes:
- This is NOT pixel blending / crossfade.
- We render one clean line-art image per frame.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

from .preprocess import load_grayscale, binarize_lineart
from .strokes import extract_strokes, resample_polyline
from .match import match_strokes, StrokeMatch
from .render import render_polylines
from .diagnostics import to_line_binary, overlap_metrics, jitter_score


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_debug_json(path: Path, data: Dict[str, Any]) -> None:
    # JSONは依存を増やさず標準で
    import json

    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_inbetween_polylines(
    strokes_a,
    strokes_b,
    matches: List[StrokeMatch],
    alpha: float,
    n_points: int,
    min_confidence: float,
) -> List[np.ndarray]:
    """
    Convert matched strokes into interpolated polylines.
    """
    polylines: List[np.ndarray] = []

    for m in matches:
        if m.confidence < min_confidence:
            continue

        sa = strokes_a[m.a_index].points
        sb = strokes_b[m.b_index].points

        # resample to same length for stable interpolation
        pa = resample_polyline(sa, n_points)
        pb = resample_polyline(sb, n_points)

        interp = (1.0 - alpha) * pa + alpha * pb
        polylines.append(interp)

    return polylines


def run_pipeline(
    image_a: Path,
    image_b: Path,
    out_dir: Path,
    inbetween_count: int = 5,
    thickness: int = 2,
    # Improvements knobs:
    n_points: int = 64,
    min_confidence: float = 0.35,
    # match weights (exposed so you can tune later)
    w_centroid: float = 1.0,
    w_len: float = 40.0,
    w_angle: float = 15.0,
    w_shape: float = 25.0,
    # save meta
    save_meta: bool = True,
) -> None:
    """
    Run improved baseline pipeline and write PNG sequence to out_dir.

    Args:
        image_a, image_b: keyframes (same resolution recommended)
        out_dir: output folder for frames
        inbetween_count: number of in-between frames
        thickness: render thickness
        n_points: polyline resample count
        min_confidence: gate low-quality matches
        w_*: matching weights
        save_meta: save JSON meta alongside frames
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # Load
    img_a = load_grayscale(image_a)
    img_b = load_grayscale(image_b)

    if img_a.shape != img_b.shape:
        raise ValueError(
            f"Keyframes must have same resolution. A={img_a.shape}, B={img_b.shape}"
        )

    # Preprocess
    bin_a = binarize_lineart(img_a)  # lines=255, bg=0
    bin_b = binarize_lineart(img_b)

    # Extract strokes
    strokes_a = extract_strokes(bin_a)
    strokes_b = extract_strokes(bin_b)

    # Match strokes (improved match.py assumed)
    matches: List[StrokeMatch] = match_strokes(
        strokes_a,
        strokes_b,
        w_centroid=w_centroid,
        w_len=w_len,
        w_angle=w_angle,
        w_shape=w_shape,
        shape_points=n_points,
    )

    # Optionally save matching summary (useful for tuning)
    if save_meta:
        summary = {
            "input": {"A": str(image_a), "B": str(image_b)},
            "params": {
                "inbetween_count": inbetween_count,
                "thickness": thickness,
                "n_points": n_points,
                "min_confidence": min_confidence,
                "match_weights": {
                    "w_centroid": w_centroid,
                    "w_len": w_len,
                    "w_angle": w_angle,
                    "w_shape": w_shape,
                },
            },
            "counts": {
                "strokes_a": len(strokes_a),
                "strokes_b": len(strokes_b),
                "matches": len(matches),
            },
            # keep it compact: top 50 by confidence
            "top_matches": [
                asdict(m) for m in sorted(matches, key=lambda x: x.confidence, reverse=True)[:50]
            ],
        }
        _save_debug_json(out_dir / "_meta_matches.json", summary)

    prev_bin: Optional[np.ndarray] = None

    for i in range(1, inbetween_count + 1):
        alpha = i / (inbetween_count + 1)

        # Build interpolated polylines
        polylines = _build_inbetween_polylines(
            strokes_a=strokes_a,
            strokes_b=strokes_b,
            matches=matches,
            alpha=alpha,
            n_points=n_points,
            min_confidence=min_confidence,
        )

        # Render single clean frame
        frame = render_polylines(polylines, img_a.shape, thickness=thickness)

        # Diagnostics
        inb_bin = to_line_binary(frame)
        om = overlap_metrics(inb_bin, bin_a, bin_b)
        jit = jitter_score(prev_bin, inb_bin)

        # Save PNG
        out_path = out_dir / f"{i:04d}.png"
        cv2.imwrite(str(out_path), frame)

        # Save per-frame meta (optional)
        if save_meta:
            _save_debug_json(
                out_dir / f"{i:04d}.json",
                {
                    "frame_index": i,
                    "alpha": float(alpha),
                    "num_polylines": len(polylines),
                    "diagnostics": {
                        "overlap_score": float(om["overlap_score"]),
                        "intersectA_ratio": float(om["intersectA_ratio"]),
                        "intersectB_ratio": float(om["intersectB_ratio"]),
                        "jitter": float(jit),
                    },
                },
            )

        print(
            f"[frame {i:04d}] polylines={len(polylines)} "
            f"overlap={om['overlap_score']:.3f} jitter={jit:.2f}"
        )

        prev_bin = inb_bin
