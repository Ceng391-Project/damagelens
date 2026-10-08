# DamageLens — satellite and aerial image analysis by hazard type

Each report: data and ground truth, methods, metric tables, charts, random-case figures, limitations and sources.
Code and raw outputs: `experiments/multi-hazard/` (one script per experiment, results in `experiments/multi-hazard/outputs/`).

![Overview](img/overview.png)

| hazard | report | platforms | headline result |
|---|---|---|---|
| Earthquake | [01-earthquake.md](01-earthquake.md) | VHR satellite (Maxar/Pleiades — Google-Earth-class) | 6 February F1 0.55; transfer from xBD F1 0.14 |
| Flood | [02-flood.md](02-flood.md) | satellite (Sentinel-1/2), VHR satellite (xBD) | 11 countries IoU 0.82; Valencia 2024 F1 0.66 |
| Landslide | [03-landslide.md](03-landslide.md) | satellite (Sentinel-2 + DEM), aerial (UAV) | satellite F1 0.63; UAV F1 0.82 |
| Tornado | [04-tornado.md](04-tornado.md) | VHR satellite (xBD), aerial (NAIP aircraft) | xBD building F1 0.76; Rolling Fork EF3+ AUC 0.57 |
| Hail | [05-hail.md](05-hail.md) | satellite (Sentinel-2) + radar MESH | ≥60 mm AUC 0.84; ≥25 mm AUC 0.61 |
| Extreme heat | [06-extreme-heat.md](06-extreme-heat.md) | satellite (MODIS LST) + stations | r 0.65; day detection AUC 0.82 |
| Fire | [07-fire.md](07-fire.md) | satellite (Sentinel-2), VHR satellite (xBD) | burned area IoU 0.54–0.86; xBD building F1 0.81 |

**About Google Earth:** Google Earth imagery cannot be downloaded in bulk or programmatically under its terms of use. Much of the post-disaster very-high-resolution imagery in Google Earth comes from Maxar; this work uses Maxar Open Data of the same class (xBD, KATE-CD, the Kahramanmaraş 2023 event). Aerial imagery was tested with UAV (landslide) and aircraft (USDA NAIP, tornado) data; no open, labelled helicopter dataset was found.
