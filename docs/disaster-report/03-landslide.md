# Landslide — Landslide4Sense (satellite) and a UAV landslide set (aerial)

## Summary
- **Satellite (Sentinel-2, 14 bands: 12 spectral + slope + DEM, 10 m), Landslide4Sense test (800 chips):** best **U-Net S2 + slope + DEM: F1 0.63**, IoU 0.46. A model using RGB only (Google-Earth-like information) reaches F1 0.58, so the gain from multispectral bands + topography is measurable.
- **The NDVI + slope rule** gets F1 0.19: not enough on its own.
- **Aerial (UAV, RGB, cm–dm resolution), 897 test chips:** **U-Net RGB (UAV): F1 0.82**, IoU 0.69; the RGB soil-index rule gets F1 0.61.
- Landslide pixels are a minority (1.9% in the satellite set, 19.1% in the UAV set), so the precision/recall balance is very threshold-sensitive.
- These two sets are single-date segmentation, not pre/post change detection; they show what the imagery can resolve, not the #30 pipeline itself.

## Data and ground truth
| set | platform | resolution | ground truth |
|---|---|---|---|
| Landslide4Sense (HF `ibm-nasa-geospatial/Landslide4sense`) | satellite: Sentinel-2 + ALOS PALSAR slope/DEM | ~10 m, 128×128 | pixel masks; 3799/245/800 train/val/test |
| UAV landslide set (HF `syeddhasnainn/landslide-uav-all`, subset) | aerial: UAV RGB | cm–dm (downsampled to 384×384) | pixel masks; 1812/431/897 used |

## Results — satellite (Landslide4Sense)
Rule thresholds were picked on validation: NDVI < -0.20 and slope > 0.99 (normalised).
| method | precision | recall | F1 | IoU |
|---|---|---|---|---|
| NDVI + slope rule | 0.107 | 0.731 | 0.187 | 0.103 |
| U-Net RGB (Google-Earth-like) | 0.579 | 0.575 | 0.577 | 0.405 |
| U-Net 12-band S2 | 0.497 | 0.700 | 0.581 | 0.410 |
| U-Net S2 + slope + DEM | 0.592 | 0.680 | 0.632 | 0.463 |

![Landslide4Sense: metrics and precision–recall curves](img/landslide_l4s_metrics.png)
*Landslide4Sense: metrics and precision–recall curves*

![Random test chips (RGB, label, rule, U-Net RGB, multispectral U-Net)](img/landslide_l4s_samples.png)
*Random test chips (RGB, label, rule, U-Net RGB, multispectral U-Net)*


## Results — aerial (UAV)
| method | precision | recall | F1 | IoU |
|---|---|---|---|---|
| RGB soil index (rule) | 0.516 | 0.745 | 0.610 | 0.439 |
| U-Net RGB (UAV) | 0.795 | 0.838 | 0.816 | 0.689 |

![UAV landslides: metrics, IoU per chip, training curve](img/landslide_uav_metrics.png)
*UAV landslides: metrics, IoU per chip, training curve*

![Random UAV test chips](img/landslide_uav_samples.png)
*Random UAV test chips*


## Limitations
- Only a subset of the UAV set was used (4 of 19 training shards); images were downsampled to 384 px.
- Landslide4Sense test regions resemble the training geography; transfer to a new region (e.g. landslides triggered by 6 February) needs its own test.

## Sources
- Landslide4Sense (Ghorbanzadeh et al., 2022): https://github.com/iarai/Landslide4Sense-2022
- UAV landslide set: https://huggingface.co/datasets/syeddhasnainn/landslide-uav-all
