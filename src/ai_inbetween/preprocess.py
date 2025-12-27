"""
preprocess.py

Image preprocessing utilities for line-art animation.
This module DOES NOT perform interpolation or inbetweening.

Responsibilities:
- Load grayscale images
- Binarize black lines on white background
- Light denoising
"""

from pathlib import Path

import cv2
import numpy as np


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load an image as grayscale (uint8)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def binarize_lineart(gray: np.ndarray) -> np.ndarray:
    """
    Convert grayscale line-art to binary image.

    Output:
      - lines: 255
      - background: 0
    """
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    # Slight blur to reduce noise from scans
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold works well for uneven paper lighting
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=5,
    )

    # Remove tiny speckles (keep lines)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return cleaned
