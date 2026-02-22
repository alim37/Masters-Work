import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def harris_corner_detection(
    gray,
    gaussian_ksize=5,
    gaussian_sigma=1.0,
    window_size=3,
    k=0.04,
    r_threshold_ratio=0.01,
    lambda_threshold=0.001,
    border_ignore_pixels=0,
):
    gray_f = gray.astype(np.float32) / 255.0
    smoothed = cv2.GaussianBlur(gray_f, (gaussian_ksize, gaussian_ksize), gaussian_sigma)

    ix = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
    iy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
    ixx = ix * ix
    iyy = iy * iy
    ixy = ix * iy

    sxx = cv2.boxFilter(ixx, ddepth=-1, ksize=(window_size, window_size), normalize=True, borderType=cv2.BORDER_CONSTANT)
    syy = cv2.boxFilter(iyy, ddepth=-1, ksize=(window_size, window_size), normalize=True, borderType=cv2.BORDER_CONSTANT)
    sxy = cv2.boxFilter(ixy, ddepth=-1, ksize=(window_size, window_size), normalize=True, borderType=cv2.BORDER_CONSTANT)

    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    r = det - k * (trace ** 2)
    discr = np.maximum(trace * trace - 4.0 * det, 0.0)
    sqrt_discr = np.sqrt(discr)
    lambda1 = 0.5 * (trace + sqrt_discr)
    lambda2 = 0.5 * (trace - sqrt_discr)

    r_thresh = r_threshold_ratio * float(np.max(r))
    lambda_thresh = float(lambda_threshold)

    corner_mask = (r > r_thresh) & (lambda1 > lambda_thresh) & (lambda2 > lambda_thresh)
    edge_mask = (lambda1 > lambda_thresh) & (lambda2 <= lambda_thresh)

    if border_ignore_pixels > 0:
        h, w = gray.shape[:2]
        valid = np.zeros((h, w), dtype=bool)
        valid[border_ignore_pixels:h - border_ignore_pixels, border_ignore_pixels:w - border_ignore_pixels] = True
        corner_mask = corner_mask & valid
        edge_mask = edge_mask & valid

    flat_mask = ~(corner_mask | edge_mask)
    ys, xs = np.where(corner_mask)
    corners_xy = np.column_stack((xs, ys)).astype(np.int32)

    return {
        "corners_xy": corners_xy,
        "corner_mask": corner_mask,
        "edge_mask": edge_mask,
        "flat_mask": flat_mask,
        "response_r": r,
        "lambda1": lambda1,
        "lambda2": lambda2,
    }


def kmeans_cluster(
    points_xy,
    k_clusters=15,
    max_iters=100,
    tol=1e-3,
    random_seed=42,
    elbow_k_min=None,
    elbow_k_max=None,
    elbow_n_init=3,
):
    points_xy = np.asarray(points_xy, dtype=np.int32)
    n_points = points_xy.shape[0]

    if n_points == 0:
        return {"labels": np.array([], dtype=np.int32), "centers": np.zeros((0, 2), dtype=np.float32)}

    if elbow_k_min is None or elbow_k_max is None:
        k_use = int(min(max(1, k_clusters), n_points))
        rng = np.random.default_rng(random_seed)
        init_idx = rng.choice(n_points, size=k_use, replace=False)
        centers = points_xy[init_idx].astype(np.float32)
        labels = np.zeros(n_points, dtype=np.int32)

        for _ in range(max_iters):
            diff = points_xy[:, None, :].astype(np.float32) - centers[None, :, :]
            dists = np.sum(diff * diff, axis=2)
            new_labels = np.argmin(dists, axis=1).astype(np.int32)

            new_centers = centers.copy()
            for c in range(k_use):
                cluster_pts = points_xy[new_labels == c]
                if cluster_pts.shape[0] > 0:
                    new_centers[c] = np.mean(cluster_pts, axis=0)
                else:
                    new_centers[c] = points_xy[rng.integers(0, n_points)]

            shift = np.linalg.norm(new_centers - centers, axis=1).max()
            labels = new_labels
            centers = new_centers
            if shift < tol:
                break

        return {"labels": labels, "centers": centers}

    k_low = max(1, int(elbow_k_min))
    k_high = min(int(elbow_k_max), n_points)
    if k_low > k_high:
        return {"ks": [], "ssd_values": []}

    ks = []
    ssd_values = []
    for k in range(k_low, k_high + 1):
        best_ssd = float("inf")
        for trial in range(elbow_n_init):
            fit = kmeans_cluster(
                points_xy,
                k_clusters=k,
                max_iters=max_iters,
                tol=tol,
                random_seed=random_seed + trial,
            )
            labels = fit["labels"]
            centers = fit["centers"]
            if centers.shape[0] == 0:
                ssd = float("nan")
            else:
                diff = points_xy.astype(np.float32) - centers[labels]
                ssd = float(np.sum(diff * diff))
            if ssd < best_ssd:
                best_ssd = ssd

        ks.append(k)
        ssd_values.append(best_ssd)

    return {"ks": ks, "ssd_values": ssd_values}


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


def plot_and_save(items, titles, out_path, mode="image"):
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, item, title in zip(axes, items, titles):
        if mode == "image":
            ax.imshow(item)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel(title, fontsize=10, labelpad=8)
        else:
            ks, ssd_vals = item
            if len(ks) == 0:
                ax.text(0.5, 0.5, "No corners", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.plot(ks, ssd_vals, marker="o", linewidth=1.8)
                ax.grid(True, alpha=0.3)
            ax.set_title(title)
            ax.set_xlabel("K")
            ax.set_ylabel("SSD")

    if mode == "elbow":
        fig.suptitle("Elbow Method on Harris Corners", fontsize=14)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    one_object_name = "plane.bmp"
    multiple_objects_name = "cars.bmp"
    no_objects_name = "castle.bmp"

    names = [one_object_name, multiple_objects_name, no_objects_name]
    images_bgr = [cv2.imread(os.path.join(".", n), cv2.IMREAD_COLOR) for n in names]

    WINDOW_SIZE = 5
    GAUSSIAN_KSIZE = 5
    GAUSSIAN_SIGMA = 1.2
    HARRIS_K = 0.04
    R_THRESHOLD_RATIO = 0.01
    LAMBDA_THRESHOLDS = [0.2, 0.2, 0.05]  # adjust per image
    BORDER_IGNORE_PIXELS = 10
    K_CLUSTERS_LIST = [10, 7, 15] # set per
    ELBOW_K_MIN = 1
    ELBOW_K_MAX = 20
    ELBOW_N_INIT = 3

    part1_outputs = []
    part2_outputs = []
    part3_outputs = []
    part_titles = []
    elbow_items = []

    for img_bgr, name, lambda_threshold, k_clusters in zip(images_bgr, names, LAMBDA_THRESHOLDS, K_CLUSTERS_LIST):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        hres = harris_corner_detection(
            gray,
            gaussian_ksize=GAUSSIAN_KSIZE,
            gaussian_sigma=GAUSSIAN_SIGMA,
            window_size=WINDOW_SIZE,
            k=HARRIS_K,
            r_threshold_ratio=R_THRESHOLD_RATIO,
            lambda_threshold=lambda_threshold,
            border_ignore_pixels=BORDER_IGNORE_PIXELS,
        )
        corners_xy = hres["corners_xy"]
        corners_overlay = draw_corners_overlay(img_bgr, corners_xy, color=(0, 0, 255), radius=2)
        part1_outputs.append(cv2.cvtColor(corners_overlay, cv2.COLOR_BGR2RGB))

        elbow = kmeans_cluster(
            corners_xy,
            max_iters=100,
            tol=1e-3,
            random_seed=42,
            elbow_k_min=ELBOW_K_MIN,
            elbow_k_max=ELBOW_K_MAX,
            elbow_n_init=ELBOW_N_INIT,
        )
        elbow_items.append((elbow["ks"], elbow["ssd_values"]))

        fit = kmeans_cluster(corners_xy, k_clusters=k_clusters, max_iters=100, tol=1e-3, random_seed=42)
        labels = fit["labels"]
        centers = fit["centers"]
        k_used = centers.shape[0]
        clusters_overlay = draw_clusters_overlay(img_bgr, corners_xy, labels, k_used)
        part2_outputs.append(cv2.cvtColor(clusters_overlay, cv2.COLOR_BGR2RGB))

        bboxes_overlay = draw_cluster_bboxes(img_bgr, corners_xy, labels, k_used, min_points_for_box=5)
        part3_outputs.append(cv2.cvtColor(bboxes_overlay, cv2.COLOR_BGR2RGB))

        part_titles.append(f"{name} | corners={len(corners_xy)} | lambda={lambda_threshold} | K={k_clusters}")

    plot_and_save(part1_outputs, part_titles, "part1_corners.png", mode="image")
    plot_and_save(part2_outputs, part_titles, "part2_clusters.png", mode="image")
    plot_and_save(part3_outputs, part_titles, "part3_bboxes.png", mode="image")
    plot_and_save(elbow_items, names, "elbow_kmeans.png", mode="elbow")

    print("Saved:")
    print(" - elbow_kmeans.png")
    print(" - part1_corners.png")
    print(" - part2_clusters.png")
    print(" - part3_bboxes.png")


if __name__ == "__main__":
    main()
