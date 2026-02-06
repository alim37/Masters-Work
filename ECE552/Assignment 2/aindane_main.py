"""
AINDANE demo / main script
Converted from MATLAB code (AINDANE_main.m).

Run:
    python aindane_main.py --image image.bmp

Notes:
- This script expects the image to be in RGB order.
- If you load with OpenCV, it loads BGR, so we convert.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage import exposure

from aindane_ale import aindane_ale
from aindane_ace import aindane_ace
from aindane_cr import aindane_cr


def ahe_per_channel_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    """MATLAB adapthisteq-like enhancement per channel (returns uint8 RGB)."""
    out = np.empty_like(img_rgb_u8)
    for ch in range(3):
        # skimage returns float in [0,1]
        eq = exposure.equalize_adapthist(img_rgb_u8[..., ch], clip_limit=0.01)
        out[..., ch] = np.clip(eq * 255.0, 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="image.bmp", help="Path to input image")
    ap.add_argument("--save_dir", type=str, default="aindane_out", help="Directory to save outputs")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load (OpenCV loads BGR)
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # --- AINDANE pipeline ---
    I_gray, I_prime = aindane_ale(rgb)

    c1, c2, c3 = 5, 20, 120
    S = aindane_ace(I_gray, I_prime, [c1, c2, c3])

    lam = 1.0
    S_f = aindane_cr(S, rgb, I_gray, lam)

    # --- Quantitative evaluation (same as MATLAB) ---
    I_zonal = I_gray[:, :]  # whole image
    S_zonal = S[:, :]
    I_z_pixel = float(I_zonal.size)

    original_mean = float(np.mean(I_gray))
    o_m = float(np.mean(I_zonal))
    original_variance = float(np.sum((o_m - I_zonal) ** 2) / I_z_pixel)
    original_sd = float(np.sqrt(original_variance))

    enhancement_mean = float(np.mean(S))
    e_m = float(np.mean(S_zonal))
    enhancement_variance = float(np.sum((e_m - S_zonal) ** 2) / I_z_pixel)
    enhancement_sd = float(np.sqrt(enhancement_variance))

    plt.figure()
    plt.plot([original_sd], [original_mean], "b+")
    plt.plot([enhancement_sd], [enhancement_mean], "r*")
    plt.plot([original_sd, enhancement_sd], [original_mean, enhancement_mean], "y")
    plt.plot([0, 91], [100, 100], "--g")
    plt.plot([35, 35], [0, 251], "--g")
    plt.axis([0, 90, 0, 250])
    plt.text(5, 170, "Insufficient Contrast")
    plt.text(5, 85, "Insufficient Contrast")
    plt.text(8, 75, "and Lightness")
    plt.text(50, 60, "Insufficient Lightness")
    plt.text(38, 230, "Vissually Optimal")
    plt.legend(["original", "enhancement"])
    plt.title("Quantitative Evaluation")
    plt.xlabel("Mean of zonal standard deviation")
    plt.ylabel("Image mean")
    plt.tight_layout()
    plt.savefig(save_dir / "quantitative_evaluation.png", dpi=200)

    # --- AHE baseline (MATLAB loop over channels) ---
    img_adjusted = ahe_per_channel_rgb(rgb)

    # --- Save outputs ---
    cv2.imwrite(str(save_dir / "original.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / "ahe.png"), cv2.cvtColor(img_adjusted, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / "aindane.png"), cv2.cvtColor(S_f, cv2.COLOR_RGB2BGR))

    # --- Visualization ---
    plt.figure()
    plt.title("Original Image")
    plt.imshow(rgb)
    plt.axis("off")

    plt.figure()
    plt.title("Enhanced by AHE")
    plt.imshow(img_adjusted)
    plt.axis("off")

    plt.figure()
    plt.title("Enhanced by AINDANE")
    plt.imshow(S_f)
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
