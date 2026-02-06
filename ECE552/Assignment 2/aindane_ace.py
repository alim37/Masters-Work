"""
AINDANE - Adaptive Contrast Enhancement (ACE)
Converted from MATLAB code (AINDANE_ACE.m).

Dependencies:
- numpy
- cv2 (OpenCV) OR scipy; here we use cv2.filter2D for speed.
"""

from __future__ import annotations
import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    cv2 = None


def _gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    """Approximate MATLAB fspecial('gaussian', ksize, sigma)."""
    if ksize <= 0:
        raise ValueError("ksize must be positive")
    if sigma <= 0:
        # Fallback: small sigma
        sigma = 0.5
    # 1D Gaussian vector then outer product
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float64)
    g = np.exp(-(ax * ax) / (2.0 * sigma * sigma))
    g /= g.sum()
    G = np.outer(g, g)
    G /= G.sum()
    return G


def aindane_ace(I: np.ndarray, I_prime: np.ndarray, multi_scale: list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
    """
    Args:
        I: grayscale float image in [0,255], shape (H,W)
        I_prime: normalized luminance in [0,1], shape (H,W)
        multi_scale: list/tuple of window sizes c (e.g., [5,20,120])

    Returns:
        S: enhanced intensity image as float64, typically in [0,255], shape (H,W)
    """
    I = I.astype(np.float64)
    I_prime = I_prime.astype(np.float64)

    scales = list(multi_scale) if np.iterable(multi_scale) else [int(multi_scale)]
    if len(scales) == 0:
        raise ValueError("multi_scale must be non-empty")

    S = np.zeros_like(I, dtype=np.float64)

    sigma_I = float(np.std(I.ravel()))
    if sigma_I <= 3:
        p = 3.0
    elif sigma_I >= 10:
        p = 1.0
    else:
        p = (27.0 - 2.0 * sigma_I) / 7.0

    for c in scales:
        c = int(c)
        if c < 1:
            continue

        gaussian_sigma = np.sqrt((c * c) / 2.0)

        # Convolution Eq.(10)
        if cv2 is None:
            # Very small fallback if cv2 isn't available
            from scipy.signal import convolve2d
            G = _gaussian_kernel_2d(c, gaussian_sigma)
            I_conv = convolve2d(I, G, mode="same", boundary="symm")
        else:
            G = _gaussian_kernel_2d(c, gaussian_sigma).astype(np.float64)
            I_conv = cv2.filter2D(I, ddepth=-1, kernel=G, borderType=cv2.BORDER_REFLECT)

        gama = I_conv / (I + 1e-4)  # avoid divide by 0
        E = gama ** p               # Eq.(11)

        S_out = 255.0 * (I_prime ** E)  # Eq.(12)
        S += S_out * (1.0 / len(scales))  # Eq.(13)

    return S
