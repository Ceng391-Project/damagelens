# DamageLens

**Disaster Damage Assessment from Satellite and Aerial Images** — damage detection from pre/post-event satellite and aerial imagery.

CENG391 Introduction to Image Understanding, 2026 Fall, term project **#30**, group G15 (3 people).

> Develop a change/damage assessment system using pre-event and post-event aerial/satellite images. Align image pairs, identify changed regions, classify or segment damaged structures/areas, and summarize damage spatially. Compare simple image-difference/feature baselines with a learned change-detection approach.

The scope is not tied to one hazard: from pre/post imagery we extract damaged **structures** (multi-class building damage) and damaged **areas** (binary masks: burned, flooded, crop damage). The main case is the **6 February 2023 Kahramanmaraş earthquakes**; the same pipeline was also tried on tornado, fire, flood and hail (`experiments/multi-hazard/`).

## Status

Feasibility is done: every step of the assignment was run end to end on open data.

| Assignment step | In code | Current result |
|---|---|---|
| Pre/post imagery | `damagelens.data` (`kate`, `xbd`, `maxar`) | KATE-CD (Maxar + Pleiades, 0.3–0.5 m), xBD, raw Maxar Open Data, NAIP aircraft imagery |
| Align image pairs | `damagelens.align` (phase correlation, ORB + RANSAC) | Raw Maxar pairs are misregistered by a median 12–16 px (~6–8 m) |
| Find changed regions, segment damage | `damagelens.models`, `damagelens.train` | KATE-CD test F1 **0.55**; xBD 5-class xView2 score **0.60** |
| Spatial summary | `damagelens.summarize`, `app/demo.py` | Kahramanmaraş centre, 1.5 km², 48 m cell heat map |
| Baselines vs learned model | `damagelens.baselines`, `damagelens.evaluate` | 5 classical methods F1 0.08–0.09, U-Net 0.55 |

![KATE-CD predictions](docs/figures/kate_predictions.png)
![Classical methods and misregistration](docs/figures/baselines_alignment.png)
![Spatial summary from raw Maxar imagery](docs/figures/spatial_summary_kahramanmaras.png)

### 6 February — KATE-CD test (38 tiles, pixel level)
| Method | F1 | IoU |
|---|---|---|
| Image difference (Lab + histogram matching) | 0.083 | 0.043 |
| CVA / 1−SSIM / PCA-kmeans / CVA+Otsu | 0.083–0.094 | 0.043–0.050 |
| U-Net 5-class trained on xBD, applied to Türkiye directly (resolution matched) | 0.139 | 0.075 |
| U-Net, xBD pretraining + KATE-CD fine-tuning | 0.500 | 0.333 |
| U-Net, KATE-CD only | 0.552 | 0.382 |
| **U-Net, KATE-CD + ±16 px shift + 200 undamaged negatives** (`--shift-px 16 --negatives 200`) | **0.579** | **0.407** |

Robustness to misregistration (pre image shifted synthetically, no registration):

| shift (px) | 0 | 8 | 16 | 32 |
|---|---|---|---|---|
| U-Net, KATE-CD only | 0.552 | 0.468 | 0.365 | 0.328 |
| U-Net, shift + negatives | 0.579 | 0.582 | 0.554 | 0.488 |

### Other hazards (summary)
Detailed reports (method, tables, charts, random cases, limitations per hazard): [docs/disaster-report/](docs/disaster-report/README.md). Single-page version: `docs/disaster-report/index.html` (open locally in a browser).

![By hazard type](docs/figures/disaster_overview.png)

| Hazard | Data | Headline result |
|---|---|---|
| Flood | Sen1Floods11 (11 countries), Valencia DANA 2024 vs Copernicus EMS | IoU 0.82; Valencia F1 0.66 |
| Landslide | Landslide4Sense (satellite), UAV set | F1 0.63 / 0.82 |
| Tornado | xBD; Rolling Fork 2023 NAIP aircraft imagery vs NWS EF points | building F1 0.76; AUC 0.57 on aerial imagery |
| Hail | Sentinel-2 ΔNDVI vs NOAA MESH, Nebraska 2022 | ≥60 mm AUC 0.84, ≥25 mm ≈0.6 |
| Extreme heat | MODIS LST vs stations, 10 cities, summer 2023 | r 0.65, day detection AUC 0.82 |
| Fire | Sentinel-2 dNBR vs NIFC/EFFIS perimeters; xBD | IoU 0.54–0.86; building F1 0.81 |

![xBD by hazard type](docs/figures/xbd_per_type.png)

## Setup

Everything in this repo runs through [uv](https://docs.astral.sh/uv/) — no manual venv, pip or requirements file. Please use uv for everything.

```bash
uv sync                        # Python 3.12 + all locked dependencies (uv.lock) into .venv
./download_data.sh kate        # ~450 MB, enough for the main experiment
./download_data.sh maxar       # STAC index for raw Maxar scenes (imagery is read on demand)
uv run pytest                  # 10 fast tests, no data needed
```

Other sets: `./download_data.sh xbd` (~24 GB), `flood`, `landslide`, `valencia`, `all`. Everything lands in `data/`. Add a dependency with `uv add <package>` (or `uv add --group experiments <package>` for experiment-only packages) and commit the updated `uv.lock`.

## Usage

```bash
uv run damagelens-train --name kate_base                                 # 6-channel U-Net on KATE-CD (~25 min on an M4)
uv run damagelens-train --name kate_robust --shift-px 16 --negatives 200  # shift augmentation + undamaged negatives
uv run damagelens-eval  --model runs/kate_base/model.pt --shifts 0 8 16 32 --register
uv run damagelens-eval  --baselines                                      # image difference, CVA, 1−SSIM, PCA-kmeans
uv run app/demo.py --model runs/kate_base/model.pt                       # raw Maxar → alignment → model → damage map
```

Outputs go to `runs/<name>/` (`model.pt` + `model.json` with the threshold and metrics). `app/demo.py` defaults to Kahramanmaraş centre; pick another area with `--lon --lat --side-m`.

### Hand labeling (raw Maxar)

```bash
uv run damagelens-label serve antakya-center   # http://127.0.0.1:8765 — regenerates missing tiles from the manifest (~1 min)
uv run damagelens-label status                 # progress
uv run damagelens-label prepare gaziantep-center --lon 37.38 --lat 37.07 --side-m 2048   # new area
```

- Pre and post side by side; click in either panel to draw a polygon, Enter closes it. Classes: `1` damaged, `2` destroyed, `3` intact building.
- Hold `B` to see the pre image in the post panel (to spot change). `V` marks a tile done, `S` skips an unusable tile, `T` jumps to the next empty tile.
- Every change is saved automatically to `labels/<area>/annotations/<tile>.json`; these are committed. Image tiles are not — they are regenerated identically from the dates in the manifest.
- When three people share an area, choose 1/3, 2/3 or 3/3 in the "share" menu so tiles do not overlap.
- Ready areas: `antakya-center` (pre 2022-12-22, closest to the event — start here) and `kahramanmaras-center` (pre 2022-07-26).
- For training: `damagelens.label.load_labeled("antakya-center")` (KATE-CD convention: damaged + destroyed = 1). For maps: `damagelens-label export <area>` → `labels.geojson`.

## Repository layout

```
src/damagelens/            the system
  data/                    KATE-CD, xBD and Maxar Open Data readers, tiling
  align/                   phase correlation, ORB + RANSAC affine alignment
  baselines/               image difference, CVA, 1−SSIM, PCA-kmeans; dNBR, NDVI, NDWI change
  models/                  6-channel U-Net (pre + post RGB)
  augment.py               flips/rotations, random pre-image shift
  train.py, evaluate.py    CLIs: damagelens-train, damagelens-eval
  summarize.py             align + predict tile by tile, per-cell damage summary
  metrics.py               pixel F1/IoU, xView2 score, building-level confusion matrix
  label/                   hand labeling: tile preparation, local web UI, labels → masks/GeoJSON
app/demo.py                end-to-end demo on a raw Maxar scene
labels/<area>/             label manifest and polygons (in git), tiles (not in git)
tests/                     fast unit tests
experiments/feasibility/   feasibility scripts (01–05), the numbers above come from these
experiments/multi-hazard/  xBD 5-class model, non-earthquake hazard tests, report builders
docs/                      README figures, hazard reports
download_data.sh           data download
```
`data/`, `runs/`, `outputs/` and model weights are not committed. `experiments/` is frozen experiment code; new work goes into `src/damagelens/`.

## Things to watch

- **Success on the labelled test does not carry over to raw scenes.** Models with F1 0.55–0.58 on KATE-CD flag only 0.6 % of pixels as damaged in heavily damaged Kahramanmaraş centre on raw Maxar imagery (0.09 % for the shift + negatives model), and less than 1 % even at threshold 0.3. The problem is neither the threshold nor alignment but a domain gap (scene selection, season, processing). This is the core problem of the project; the fix is fine-tuning on some labelled raw Maxar data (#2, #3).
- **Every KATE-CD tile contains damage.** The model never saw an undamaged scene and the false-alarm rate cannot be measured. Undamaged negatives must be added to training.
- **KATE-CD has no coordinates,** so the spatial summary needs georeferenced ground truth to be validated. We found no other open building-level set for Türkiye: the HOT OSM export was empty and Copernicus EMS EMSR648 requires a login.
- **Season, viewing angle, snow and cloud:** many pre images are summer 2022, the post images February 2023. Snow and cloud produce false damage in both models. Prefer December 2022 / January 2023 pre images where available (e.g. Antakya).
- **Alignment:** a 16 px shift drops F1 from 0.55 to 0.36. Phase correlation brings it back to 0.49; tall buildings have parallax, so a single shift is not enough.
- **Learning from other disasters does not transfer:** xBD applied to Türkiye directly gets F1 0.11–0.14, and xBD pretraining did not help fine-tuning either.
- **Splits must be deterministic.** The first xBD run used a tier3 split that changed with Python's per-process hash seed, so evaluation overlapped with training (xView2 0.62 instead of 0.60). Fixed in #21; never iterate a `set` when assigning splits.
- **Small test set (38 tiles):** report results as mean ± std over 3 seeds or k-fold.
- **Licences:** xBD is CC BY-NC-SA 4.0 and Maxar Open Data CC BY-NC 4.0, both non-commercial only. The KATE-CD licence is not stated; ask the authors (#4).

## Roadmap

Work is tracked as issues on the [DamageLens project board](https://github.com/orgs/Ceng391-Project/projects/1); each issue has a linked `feat/<no>-<name>` branch. Role labels: **role: A data & alignment**, **role: B model**, **role: C evaluation**.

- Priority: #1 labeling tool, #2 raw Maxar label set, #3 fine-tuning and raw-scene evaluation
- A: #4 KATE-CD licence, #5 ORB vs phase correlation, #6 snow/cloud mask, #7 real negatives
- B: #8 colour/shadow augmentation, #9 Siamese model, #10 full-resolution xBD + focal loss
- C: #11 seeds / k-fold, #12 district-level summary + map, #13 final report and presentation
- Repo: #14 uv-only setup, #15 English translation

## Data sources

- KATE-CD: https://huggingface.co/datasets/CSCRS/kate-cd (ITÜ CSCRS)
- xBD / xView2: https://xview2.org, mirror https://huggingface.co/datasets/hannan022/xview2-xbd
- Maxar Open Data Program: https://maxar-opendata.s3.amazonaws.com/events/catalog.json
- Sen1Floods11, Landslide4Sense, UAV landslide set, Copernicus EMS EMSR773, NOAA MRMS/SPC/NWS DAT, MODIS LST, Meteostat, NIFC, EFFIS: details in the `experiments/multi-hazard/` scripts
