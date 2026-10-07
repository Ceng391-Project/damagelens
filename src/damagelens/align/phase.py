import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.color import rgb2gray
from skimage.registration import phase_cross_correlation


def estimate_shift(reference: np.ndarray, moving: np.ndarray, upsample: int = 4) -> np.ndarray:
    # returns the (dy, dx) that moves `moving` onto `reference`
    return phase_cross_correlation(rgb2gray(reference), rgb2gray(moving), upsample_factor=upsample)[0]


def apply_shift(img: np.ndarray, shift: np.ndarray) -> np.ndarray:
    s = (shift[0], shift[1]) + (0,) * (img.ndim - 2)
    return np.clip(nd_shift(img.astype(np.float32), s, order=1, mode="nearest"), 0, 255).astype(img.dtype)


def register_phase(reference: np.ndarray, moving: np.ndarray, max_shift: float = 40.0):
    # shifts above max_shift are almost always a wrong correlation peak, not real misregistration
    sh = estimate_shift(reference, moving)
    if float(np.hypot(*sh)) > max_shift:
        return moving, np.zeros(2), False
    return apply_shift(moving, sh), sh, True
