# Fire — burned area (Sentinel-2 dNBR) and building damage (xBD)

## Summary
- **Burned area, 4 fires, against official perimeters:** Palisades IoU 0.86, Eaton IoU 0.72, Camp IoU 0.54, Manavgat IoU 0.72 (best threshold).
- A single standard threshold (USGS dNBR > 0.10) works well without training on most fires; errors come from unburned islands inside the official perimeter, harvested fields and cloud.
- **Fire-driven building damage (xBD, 5 fire events):** damaged/undamaged building F1 **0.81**, destroyed-building recall 0.79 — fire is one of the best-detected hazards in xBD (burned buildings are binary: standing or ash).

## Data and ground truth
| data | platform | resolution | ground truth |
|---|---|---|---|
| Sentinel-2 L2A B08/B12 pre/post mosaics — Planetary Computer | satellite | 20 m (~25 m grid) | NIFC WFIGS / InterAgency fire perimeters (US), EFFIS burned area (Manavgat) |
| xBD: socal, santa-rosa, woolsey, portugal, pinery | satellite: Maxar | ~0.5 m (~1 m in the model) | building damage levels |

## Results — burned area
| fire | method | precision | recall | F1 | IoU | official area km² | satellite area km²** |
|---|---|---|---|---|---|---|---|
| Palisades (LA, Jan 2025) | dNBR>0.10 (USGS low) | 0.849 | 0.967 | 0.904 | 0.825 | 96.352 | 106.656 |
| Palisades (LA, Jan 2025) | dNBR>0.27 (USGS moderate-low) | 0.987 | 0.860 | 0.919 | 0.851 | 96.352 | 81.654 |
| Palisades (LA, Jan 2025) | Otsu (0.26) | 0.986 | 0.867 | 0.922 | 0.856 | 96.352 | 82.379 |
| Eaton (LA, Jan 2025) | dNBR>0.10 (USGS low) | 0.755 | 0.939 | 0.837 | 0.720 | 56.882 | 52.229 |
| Eaton (LA, Jan 2025) | dNBR>0.27 (USGS moderate-low) | 0.974 | 0.694 | 0.810 | 0.681 | 56.882 | 29.913 |
| Eaton (LA, Jan 2025) | Otsu (0.24) | 0.964 | 0.743 | 0.839 | 0.723 | 56.882 | 32.352 |
| Camp (Paradise, Nov 2018) | dNBR>0.10 (USGS low) | 0.631 | 0.792 | 0.702 | 0.541 | 621.322 | 780.138 |
| Camp (Paradise, Nov 2018) | dNBR>0.27 (USGS moderate-low) | 0.833 | 0.551 | 0.663 | 0.496 | 621.322 | 410.582 |
| Camp (Paradise, Nov 2018) | Otsu (0.27) | 0.834 | 0.549 | 0.662 | 0.495 | 621.322 | 408.195 |
| Manavgat (Antalya, Jul 2021) | dNBR>0.10 (USGS low) | 0.824 | 0.853 | 0.839 | 0.722 | 548.800 | 568.184 |
| Manavgat (Antalya, Jul 2021) | dNBR>0.27 (USGS moderate-low) | 0.909 | 0.625 | 0.741 | 0.589 | 548.800 | 377.419 |
| Manavgat (Antalya, Jul 2021) | Otsu (0.28) | 0.913 | 0.603 | 0.726 | 0.570 | 548.800 | 362.524 |

\*\* The satellite area is measured over the whole perimeter box with a 25 % buffer (other burns inside the box count too).

![IoU, area comparison and burn-severity mix inside the perimeter](img/fire_metrics.png)
*IoU, area comparison and burn-severity mix inside the perimeter*

![NBR pre/post, dNBR and error map](img/fire_maps.png)
*NBR pre/post, dNBR and error map*


## Results — building damage (xBD fire events)
**Model:** 6-channel (pre + post RGB) U-Net / ResNet18, 5 classes (background, no damage, minor, major, destroyed). Trained on ≤250 images per event from xBD train + tier3 (3304 images), downsampled 1024 → 512 (~1 m/px), 12 epochs; best validation xView2 score 0.590. Test: xBD test split + a held-out 20 % of the tier3 events (1313 images). *Damage class F1* is the harmonic mean over the four damage classes (xView2 definition); *building* metrics use connected building components.

| event | buildings | building localisation F1 | damage class F1 | damaged/undamaged F1 (building) | destroyed recall | true damaged share |
|---|---|---|---|---|---|---|
| pinery-bushfire | 242 | 0.619 | 0.000 | 0.386 | 0.360 | 0.169 |
| portugal-wildfire | 752 | 0.689 | 0.000 | 0.333 | 0.192 | 0.048 |
| santa-rosa-wildfire | 3833 | 0.748 | 0.023 | 0.921 | 0.934 | 0.247 |
| socal-fire | 3575 | 0.697 | 0.106 | 0.634 | 0.580 | 0.131 |
| woolsey-fire | 617 | 0.695 | 0.000 | 0.741 | 0.642 | 0.311 |
| **fire (all)** | 9019 | 0.717 | 0.061 | 0.807 | 0.787 | 0.186 |

![xBD fire events — random test tiles](img/fire_xbd_samples.png)
*xBD fire events — random test tiles*


## Limitations
- Official perimeters include unburned islands, so the IoU ceiling is below 1.
- The EFFIS perimeter is MODIS/VIIRS + Sentinel-2 based, the NIFC perimeters are airborne/field mapped — consistency differs between sources.
- For the Camp fire the December "post" mosaic has cloud/snow and harvested fields that produce false alarms.

## Sources
- NIFC WFIGS Interagency Perimeters: https://data-nifc.opendata.arcgis.com
- EFFIS: https://effis.jrc.ec.europa.eu
- xBD / xView2 (Gupta et al., 2019), Maxar Open Data imagery, CC BY-NC-SA 4.0 — HF mirror `hannan022/xview2-xbd`
