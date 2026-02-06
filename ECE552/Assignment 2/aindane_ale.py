"""
AINDANE - Adaptive Luminance Enhancement (ALE)
Converted from MATLAB code (AINDANE_ALE.m).

Dependencies:
- numpy
"""

from __future__ import annotations
import numpy as np


def aindane_ale(image_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image_u8: uint8 image, shape (H,W) or (H,W,3) in RGB order.

    Returns:
        I: grayscale intensity image as float64 in [0,255], shape (H,W)
        In_prime: enhanced normalized luminance as float64 in [0,1], shape (H,W)
    """
    if image_u8.ndim == 3 and image_u8.shape[2] == 3:
        # MATLAB uses RGB; coefficients correspond to NTSC: 0.299, 0.587, 0.114
        Ir = image_u8[..., 0].astype(np.float64)
        Ig = image_u8[..., 1].astype(np.float64)
        Ib = image_u8[..., 2].astype(np.float64)
        I = (76.245 * Ir + 149.685 * Ig + 29.07 * Ib) / 255.0  # Eq.(1)
    else:
        I = image_u8.astype(np.float64)

    # Normalization Eq.(2)
    In = I / 255.0

    # Compute CDF similarly to MATLAB histcounts(...,255)
    # Use 255 bins over [0, 255] and include 255 with a tiny epsilon.
    y, _ = np.histogram(I.ravel(), bins=255, range=(0.0, 255.0 + 1e-9))
    total_pixels = float(y.sum()) if y.sum() > 0 else 1.0
    cdf_image = np.cumsum(y) / total_pixels

    # Find L where cdf is closest to 0.1 (MATLAB index is 1..255)
    L = int(np.argmin(np.abs(cdf_image - 0.1)) + 1)

    # Compute z
    if L <= 50:
        z = 0.0
    elif L > 150:
        z = 1.0
    else:
        z = (L - 50.0) / 100.0

    # Compute In' Eq.(3)
    In_prime = (
        0.5 * (In ** (0.75 * z + 0.25))
        + 0.2 * (1.0 - In) * (1.0 - z)
        + 0.5 * (In ** (2.0 - z))
    )

    return I, In_prime
