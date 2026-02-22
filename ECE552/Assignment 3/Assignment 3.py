import os
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class HarrisResult:
    corners_xy: np.ndarray
    corner_mask: np.ndarray
    edge_mask: np.ndarray
    flat_mask: np.ndarray
    response_r: np.ndarray
    lambda1: np.ndarray
    lambda2: np.ndarray


def load_three_bmp_images(folder="."):
    one_object_name = "plane.bmp"
    multiple_objects_name = "cars.bmp"
    no_objects_name = "castle.bmp"

    chosen = [
        os.path.join(folder, one_object_name),
        os.path.join(folder, multiple_objects_name),
        os.path.join(folder, no_objects_name),
    ]
    images = []
    names = []
    for p in chosen:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Could not read image: {p}")
        images.append(img_bgr)
        names.append(os.path.basename(p))
    return images, names


def harris_corner_detection(gray, gaussian_ksize=5, gaussian_sigma=1.0, window_size=3, k=0.04, r_threshold_ratio=0.01, lambda_threshold=1000.0, border_ignore_pixels=0):
    
    gray_f = gray.astype(np.float32)

    # 1) Gaussian smoothing to reduce noise-driven corners.
    smoothed = cv2.GaussianBlur(gray_f, (gaussian_ksize, gaussian_ksize), gaussian_sigma)

    # 2) Sobel gradients with zero padding at image borders.
    ix = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
    iy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)

    # 3) Elements of structure tensor C over local n x n window.
    ixx = ix * ix
    iyy = iy * iy
    ixy = ix * iy

    # 4) Average over n x n region around each pixel (tunable window_size).
    sxx = cv2.boxFilter(ixx, ddepth=-1, ksize=(window_size, window_size), normalize=True, borderType=cv2.BORDER_CONSTANT)
    syy = cv2.boxFilter(iyy, ddepth=-1, ksize=(window_size, window_size), normalize=True, borderType=cv2.BORDER_CONSTANT)
    sxy = cv2.boxFilter(ixy, ddepth=-1, ksize=(window_size, window_size), normalize=True, borderType=cv2.BORDER_CONSTANT)

    # 5) Eigen analysis of C at each pixel:
    # C = [[sxx, sxy], [sxy, syy]]
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    r = det - k * (trace ** 2)
    discr = np.maximum(trace * trace - 4.0 * det, 0.0)
    sqrt_discr = np.sqrt(discr)
    lambda1 = 0.5 * (trace + sqrt_discr)
    lambda2 = 0.5 * (trace - sqrt_discr)

    # 6) Thresholds (tunable).
    r_thresh = r_threshold_ratio * float(np.max(r))
    lambda_thresh = float(lambda_threshold)

    # 7) If-else style logic (vectorized): corner / edge / flat
    corner_mask = (r > r_thresh) & (lambda1 > lambda_thresh) & (lambda2 > lambda_thresh)
    edge_mask = (lambda1 > lambda_thresh) & (lambda2 <= lambda_thresh)

    # Ignore border pixels (common source of false corners due to padding).
    if border_ignore_pixels > 0:
        h, w = gray.shape[:2]
        valid = np.zeros((h, w), dtype=bool)
        valid[border_ignore_pixels:h - border_ignore_pixels, border_ignore_pixels:w - border_ignore_pixels] = True
        corner_mask = corner_mask & valid
        edge_mask = edge_mask & valid

    flat_mask = ~(corner_mask | edge_mask)

    ys, xs = np.where(corner_mask)
    corners_xy = np.column_stack((xs, ys)).astype(np.int32)

    return HarrisResult(
        corners_xy=corners_xy,
        corner_mask=corner_mask,
        edge_mask=edge_mask,
        flat_mask=flat_mask,
        response_r=r,
        lambda1=lambda1,
        lambda2=lambda2,
    )


def kmeans_points(points_xy, k_clusters=15, max_iters=100, tol=1e-3, random_seed=42):
    if points_xy.shape[0] == 0:
        return np.array([], dtype=np.int32), np.zeros((0, 2), dtype=np.float32)

    n_points = points_xy.shape[0]
    k_use = int(min(k_clusters, n_points))
    rng = np.random.default_rng(random_seed)

    # 1) Randomly choose K points as initial cluster means.
    init_idx = rng.choice(n_points, size=k_use, replace=False)
    centers = points_xy[init_idx].astype(np.float32)
    labels = np.zeros(n_points, dtype=np.int32)

    for _ in range(max_iters):
        # 2) Euclidean distance to every cluster mean.
        diff = points_xy[:, None, :].astype(np.float32) - centers[None, :, :]
        dists = np.sum(diff * diff, axis=2)

        # 3) Assign each point to nearest cluster mean.
        new_labels = np.argmin(dists, axis=1).astype(np.int32)

        # 4/5) Recompute means.
        new_centers = centers.copy()
        for c in range(k_use):
            cluster_pts = points_xy[new_labels == c]
            if cluster_pts.shape[0] > 0:
                new_centers[c] = np.mean(cluster_pts, axis=0)
            else:
                new_centers[c] = points_xy[rng.integers(0, n_points)]

        # 6) Repeat until means are stable.
        shift = np.linalg.norm(new_centers - centers, axis=1).max()
        labels = new_labels
        centers = new_centers
        if shift < tol:
            break

    return labels, centers


def draw_corners_overlay(img_bgr, corners_xy, color=(0, 0, 255), radius=2):
    out = img_bgr.copy()
    for x, y in corners_xy:
        cv2.circle(out, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def draw_clusters_overlay(img_bgr, points_xy, labels, k_clusters):
    out = img_bgr.copy()
    if points_xy.shape[0] == 0:
        return out
    cmap = plt.colormaps.get_cmap("tab20")
    for idx, (x, y) in enumerate(points_xy):
        c = cmap((int(labels[idx]) % 20) / 19 if 19 > 0 else 0.0)
        bgr = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
        cv2.circle(out, (int(x), int(y)), 2, bgr, -1, lineType=cv2.LINE_AA)
    return out


def draw_cluster_bboxes(img_bgr, points_xy, labels, k_clusters, min_points_for_box=5):
    out = img_bgr.copy()
    if points_xy.shape[0] == 0:
        return out
    cmap = plt.colormaps.get_cmap("tab20")
    for c in range(k_clusters):
        cluster_pts = points_xy[labels == c]
        if cluster_pts.shape[0] < min_points_for_box:
            continue
        x_min, y_min = np.min(cluster_pts, axis=0).astype(int)
        x_max, y_max = np.max(cluster_pts, axis=0).astype(int)
        cc = cmap((c % 20) / 19 if 19 > 0 else 0.0)
        bgr = (int(cc[2] * 255), int(cc[1] * 255), int(cc[0] * 255))
        cv2.rectangle(out, (x_min, y_min), (x_max, y_max), bgr, 2, lineType=cv2.LINE_AA)
    return out


def plot_and_save(images_rgb, titles, suptitle, out_path):
    n = len(images_rgb)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, im, title in zip(axes, images_rgb, titles):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    images_bgr, names = load_three_bmp_images(".")

    # Tunable parameters
    WINDOW_SIZE = 5
    GAUSSIAN_KSIZE = 5
    GAUSSIAN_SIGMA = 1.2
    HARRIS_K = 0.04
    R_THRESHOLD_RATIO = 0.01
    LAMBDA_THRESHOLD = 1000.0
    # Increase this value if image-frame corners still appear.
    BORDER_IGNORE_PIXELS = 10
    K_CLUSTERS = 15

    part1_outputs = []
    part2_outputs = []
    part3_outputs = []
    part_titles = []

    for img_bgr, name in zip(images_bgr, names):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Part1
        hres = harris_corner_detection(
            gray,
            gaussian_ksize=GAUSSIAN_KSIZE,
            gaussian_sigma=GAUSSIAN_SIGMA,
            window_size=WINDOW_SIZE,
            k=HARRIS_K,
            r_threshold_ratio=R_THRESHOLD_RATIO,
            lambda_threshold=LAMBDA_THRESHOLD,
            border_ignore_pixels=BORDER_IGNORE_PIXELS,
        )
        corners_overlay = draw_corners_overlay(img_bgr, hres.corners_xy, color=(0, 0, 255), radius=2)
        part1_outputs.append(cv2.cvtColor(corners_overlay, cv2.COLOR_BGR2RGB))

        # Part2
        labels, centers = kmeans_points(hres.corners_xy, k_clusters=K_CLUSTERS, max_iters=100, tol=1e-3, random_seed=42)
        k_used = centers.shape[0]
        clusters_overlay = draw_clusters_overlay(img_bgr, hres.corners_xy, labels, k_used)
        part2_outputs.append(cv2.cvtColor(clusters_overlay, cv2.COLOR_BGR2RGB))

        # Part3
        bboxes_overlay = draw_cluster_bboxes(
            img_bgr,
            hres.corners_xy,
            labels,
            k_used,
            min_points_for_box=5,
        )
        part3_outputs.append(cv2.cvtColor(bboxes_overlay, cv2.COLOR_BGR2RGB))

        part_titles.append(f"{name} | corners={len(hres.corners_xy)} | clusters={k_used}")

    # Save 3 figures
    plot_and_save(part1_outputs, part_titles, "Part 1: Harris Corner Detection", "part1_corners.png")
    plot_and_save(part2_outputs, part_titles, "Part 2: K-means Clustering on Corners", "part2_clusters.png")
    plot_and_save(part3_outputs, part_titles, "Part 3: Cluster Bounding Boxes", "part3_bboxes.png")

    print("Saved:")
    print(" - part1_corners.png")
    print(" - part2_clusters.png")
    print(" - part3_bboxes.png")


if __name__ == "__main__":
    main()
