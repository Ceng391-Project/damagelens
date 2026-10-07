import numpy as np
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform, warp


def register_orb(reference: np.ndarray, moving: np.ndarray, n_keypoints: int = 1500, min_inliers: int = 12):
    feats = []
    for img in (reference, moving):
        orb = ORB(n_keypoints=n_keypoints, fast_threshold=0.05)
        orb.detect_and_extract(rgb2gray(img))
        feats.append((orb.keypoints, orb.descriptors))
    (k_ref, d_ref), (k_mov, d_mov) = feats
    matches = match_descriptors(d_ref, d_mov, cross_check=True, max_ratio=0.8)
    if len(matches) < min_inliers:
        return moving, None, 0, False
    src = k_mov[matches[:, 1]][:, ::-1]  # (row, col) → (x, y)
    dst = k_ref[matches[:, 0]][:, ::-1]
    model, inliers = ransac((src, dst), AffineTransform, min_samples=3, residual_threshold=2.0, max_trials=2000, rng=0)
    n = int(inliers.sum()) if inliers is not None else 0
    if model is None or n < min_inliers:
        return moving, None, n, False
    aligned = warp(moving, model.inverse, output_shape=reference.shape[:2], order=1, mode="edge", preserve_range=True)
    return aligned.astype(moving.dtype), model, n, True
