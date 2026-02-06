"""
AINDANE - Color Restoration (CR)
Converted from MATLAB code (AINDANE_CR.m).

Dependencies:
- numpy
"""

from __future__ import annotations
import numpy as np


def aindane_cr(S: np.ndarray, I_input_u8: np.ndarray, I_gray: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """
    Args:
        S: enhanced grayscale intensity (float), shape (H,W)
        I_input_u8: original uint8 image, shape (H,W) or (H,W,3) RGB
        I_gray: grayscale intensity (float) in [0,255], shape (H,W)
        lam: lambda scaling factor

    Returns:
        uint8 restored image, same shape as input
    """
    S = S.astype(np.float64)
    I_gray = I_gray.astype(np.float64)

    if I_input_u8.ndim == 3 and I_input_u8.shape[2] == 3:
        lr = 0.99 * lam
        lg = 0.99 * lam
        lb = 0.99 * lam

        Ir = I_input_u8[..., 0].astype(np.float64)
        Ig = I_input_u8[..., 1].astype(np.float64)
        Ib = I_input_u8[..., 2].astype(np.float64)

        denom = (I_gray + 1e-4)
        Sr = S * Ir / denom * lr
        Sg = S * Ig / denom * lg
        Sb = S * Ib / denom * lb

        S_t = np.stack([Sr, Sg, Sb], axis=2)
        S_t = np.clip(S_t, 0, 255).astype(np.uint8)
        return S_t
    else:
        out = np.clip(S * lam, 0, 255).astype(np.uint8)
        return out
