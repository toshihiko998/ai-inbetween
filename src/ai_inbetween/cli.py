"""
cli.py

Command-line interface for ai-inbetween.
"""

from pathlib import Path
import argparse

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="AI In-between (baseline)")
    parser.add_argument("--a", required=True, help="path to keyframe A")
    parser.add_argument("--b", required=True, help="path to keyframe B")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--n", type=int, default=5, help="number of in-betweens")
    parser.add_argument("--thickness", type=int, default=2, help="line thickness")

    args = parser.parse_args()

    run_pipeline(
        image_a=Path(args.a),
        image_b=Path(args.b),
        out_dir=Path(args.out),
        inbetween_count=args.n,
        thickness=args.thickness,
    )


if __name__ == "__main__":
    main()
