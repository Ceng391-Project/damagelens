import numpy as np
from skimage.color import rgb2gray, rgb2lab
from skimage.exposure import match_histograms
from skimage.filters import gaussian
from skimage.metrics import structural_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def image_difference(pre, post, sigma=3):
    pre = match_histograms(pre, post, channel_axis=-1)
    return gaussian(np.linalg.norm(rgb2lab(post) - rgb2lab(pre), axis=-1), sigma)


cva = image_difference  # change vector analysis on Lab = magnitude of the colour change vector


def ssim_change(pre, post, sigma=3, win=11):
    _, s = structural_similarity(rgb2gray(pre), rgb2gray(post), full=True, data_range=1.0, win_size=win)
    return gaussian(1 - s, sigma)


def pca_kmeans(pre, post, h=4, n_components=3, sample=17):
    # Celik (2009), IEEE GRSL 6(4)
    d = np.abs(rgb2gray(post) - rgb2gray(match_histograms(pre, post, channel_axis=-1)))
    blocks = np.lib.stride_tricks.sliding_window_view(np.pad(d, h, mode="reflect"), (2 * h + 1, 2 * h + 1)).reshape(d.size, -1)
    f = PCA(n_components).fit(blocks[::sample]).transform(blocks)
    lab = KMeans(2, n_init=3, random_state=0).fit(f[::sample]).predict(f).reshape(d.shape)
    hi = int(np.argmax([d[lab == k].mean() for k in range(2)]))
    return gaussian((lab == hi).astype(float), 1)
