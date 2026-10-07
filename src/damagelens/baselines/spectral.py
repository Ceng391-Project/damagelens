import numpy as np


def _ratio(a, b):
    a, b = np.clip(a, 0, None), np.clip(b, 0, None)
    return np.clip((a - b) / (a + b + 1e-3), -1, 1)


def nbr(nir, swir2):
    return _ratio(nir, swir2)


def dnbr(pre_nir, pre_swir2, post_nir, post_swir2):
    # USGS severity classes: > 0.10 low, > 0.27 moderate-low, > 0.44 moderate-high, > 0.66 high
    return nbr(pre_nir, pre_swir2) - nbr(post_nir, post_swir2)


def ndvi(red, nir):
    return _ratio(nir, red)


def ndwi(green, nir):
    return _ratio(green, nir)
