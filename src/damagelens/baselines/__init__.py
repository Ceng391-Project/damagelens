from .image_diff import cva, image_difference, pca_kmeans, ssim_change
from .spectral import dnbr, nbr, ndvi, ndwi

BASELINES = {"image_difference": image_difference, "cva": cva, "ssim": ssim_change, "pca_kmeans": pca_kmeans}

__all__ = ["BASELINES", "image_difference", "cva", "ssim_change", "pca_kmeans", "nbr", "dnbr", "ndvi", "ndwi"]
