"""
AINDANE batch demo / main script

Runs AINDANE + AHE on a list of input BMPs and saves:
- original_{i}.png
- ahe_{i}.png
- aindane_{i}.png
- quantitative_evaluation_{i}.png        (Original vs AINDANE metric plot)
- quantitative_evaluation_ahe_{i}.png    (Original vs AHE metric plot)

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
        eq = exposure.equalize_adapthist(img_rgb_u8[..., ch], clip_limit=0.01)  # float [0,1]
        out[..., ch] = np.clip(eq * 255.0, 0, 255).astype(np.uint8)
    return out


def plot_quantitative_eval(
    *,
    I_gray: np.ndarray,
    enhanced_gray: np.ndarray,
    save_path: Path,
    label_enh: str,
    title: str,
):
    """Replicates the MATLAB-style quantitative evaluation plot."""
    I_gray = I_gray.astype(np.float64)
    enhanced_gray = enhanced_gray.astype(np.float64)

    I_zonal = I_gray[:, :]
    E_zonal = enhanced_gray[:, :]
    I_z_pixel = float(I_zonal.size)

    original_mean = float(np.mean(I_gray))
    o_m = float(np.mean(I_zonal))
    original_variance = float(np.sum((o_m - I_zonal) ** 2) / I_z_pixel)
    original_sd = float(np.sqrt(original_variance))

    enh_mean = float(np.mean(enhanced_gray))
    e_m = float(np.mean(E_zonal))
    enh_variance = float(np.sum((e_m - E_zonal) ** 2) / I_z_pixel)
    enh_sd = float(np.sqrt(enh_variance))

    plt.figure()
    plt.plot([original_sd], [original_mean], "b+")
    plt.plot([enh_sd], [enh_mean], "r*")
    plt.plot([original_sd, enh_sd], [original_mean, enh_mean], "y")
    plt.plot([0, 91], [100, 100], "--g")
    plt.plot([35, 35], [0, 251], "--g")
    plt.axis([0, 90, 0, 250])
    plt.text(5, 170, "Insufficient Contrast")
    plt.text(5, 85, "Insufficient Contrast")
    plt.text(8, 75, "and Lightness")
    plt.text(50, 60, "Insufficient Lightness")
    plt.text(38, 230, "Vissually Optimal")
    plt.legend(["original", label_enh])
    plt.title(title)
    plt.xlabel("Mean of zonal standard deviation")
    plt.ylabel("Image mean")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def process_one_image(img_path: Path, save_dir: Path, idx: int):
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

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

    # --- AHE baseline (per-channel) ---
    img_adjusted = ahe_per_channel_rgb(rgb)

    # --- Save outputs ---
    cv2.imwrite(str(save_dir / f"original_{idx}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / f"ahe_{idx}.png"), cv2.cvtColor(img_adjusted, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_dir / f"aindane_{idx}.png"), cv2.cvtColor(S_f, cv2.COLOR_RGB2BGR))

    # --- Quantitative evaluation: Original vs AINDANE ---
    plot_quantitative_eval(
        I_gray=I_gray,
        enhanced_gray=S,  # grayscale enhancement metric uses S (same as your original script)
        save_path=save_dir / f"quantitative_evaluation_{idx}.png",
        label_enh="enhancement",
        title="Quantitative Evaluation",
    )

    # --- Quantitative evaluation: Original vs AHE ---
    ahe_gray = cv2.cvtColor(img_adjusted, cv2.COLOR_RGB2GRAY)
    plot_quantitative_eval(
        I_gray=I_gray,
        enhanced_gray=ahe_gray,
        save_path=save_dir / f"quantitative_evaluation_ahe_{idx}.png",
        label_enh="AHE",
        title="Quantitative Evaluation (AHE)",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images",
        nargs="+",
        default=["1.bmp", "2.bmp", "3.bmp", "4.bmp", "5.bmp", "6.bmp"],
        help="List of input images (BMP)",
    )
    ap.add_argument("--save_dir", type=str, default="aindane_out", help="Directory to save outputs")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(args.images, start=1):
        process_one_image(Path(img), save_dir, i)


if __name__ == "__main__":
    main()
