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
    """Load image as grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def binarize_lineart(gray: np.ndarray) -> np.ndarray:
    """
    Convert grayscale line-art to binary image.
    Lines -> 255, Background -> 0
    """
    # Slight blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=5,
    )

    # Remove tiny speckles
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned
