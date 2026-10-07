import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).parent
O = ROOT / "outputs"
F = ROOT.parent / "feasibility" / "outputs"
R = ROOT.parent.parent / "docs" / "disaster-report"
IMG = R / "img"
IMG.mkdir(parents=True, exist_ok=True)
J = lambda p: json.load(open(p))


def img(src, name, caption):
    shutil.copy(src, IMG / name)
    return f"![{caption}](img/{name})\n*{caption}*\n"


def table(header, rows):
    out = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        out += "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |\n"
    return out


xe = J(O / "xbd/eval_xbd_unet5.json")
xtl = J(O / "xbd/train_log.json")


def xbd_rows(types):
    rows = []
    for ev, m in sorted(xe["per_event"].items()):
        if m["type"] in types:
            rows.append([ev, m["buildings"], m["loc_f1"], m["dmg_f1"], m["building_damaged_f1"], m["destroyed_recall"], m["gt_damaged_share"]])
    for t in types:
        if t in xe["per_type"]:
            m = xe["per_type"][t]
            rows.append([f"**{t} (all)**", m["buildings"], m["loc_f1"], m["dmg_f1"], m["building_damaged_f1"], m["destroyed_recall"], m["gt_damaged_share"]])
    return table(["event", "buildings", "building localisation F1", "damage class F1", "damaged/undamaged F1 (building)", "destroyed recall", "true damaged share"], rows)


XBD_NOTE = (f"**Model:** 6-channel (pre + post RGB) U-Net / ResNet18, 5 classes (background, no damage, minor, major, destroyed). "
            f"Trained on ≤250 images per event from xBD train + tier3 ({xtl['n_train']} images), downsampled 1024 → 512 (~1 m/px), 12 epochs; "
            f"best validation xView2 score {max(h['val_xview2'] for h in xtl['hist']):.3f}. Test: xBD test split + a held-out 20 % of the tier3 events "
            f"({xe['n_test_images']} images). *Damage class F1* is the harmonic mean over the four damage classes (xView2 definition); *building* metrics use connected building components.\n")
SRC_XBD = "- xBD / xView2 (Gupta et al., 2019), Maxar Open Data imagery, CC BY-NC-SA 4.0 — HF mirror `hannan022/xview2-xbd`\n"
reports = {}

# ---------------- EARTHQUAKE ----------------
fb = J(F / "01_stats_baseline.json"); k1, k2, k3 = J(F / "kate_only.json"), J(F / "xbd_only.json"), J(F / "xbd_then_kate.json")
kz = xe["kate_zero_shot"]; mx = J(O / "earthquake_maxar/results.json")
best_kz = max(kz.values(), key=lambda v: v["f1"])
mex = xe["per_event"].get("mexico-earthquake", {}); palu = xe["per_event"].get("palu-tsunami", {})
reports["01-earthquake"] = f"""# Earthquake — 6 February 2023 Kahramanmaraş and the xBD earthquakes

## Summary
- **6 February (KATE-CD, Maxar + Pleiades, 0.3–0.5 m — Google-Earth-class imagery):** a model trained on Türkiye data reaches **F1 {k1['test']['f1']:.2f}** (IoU {k1['test']['iou']:.2f}) on the test split; plain image differencing gets {fb['baseline_test']['f1']:.2f}.
- **Learning from other earthquakes does not transfer to Türkiye:** the 5-class model trained on xBD (Mexico 2017, Palu 2018 and other disasters) scores **F1 {best_kz['f1']:.2f}** on 6 February without fine-tuning.
- **Earthquake/tsunami events inside xBD:** building localisation F1 {mex.get('loc_f1', float('nan')):.2f} for the Mexico earthquake, but only {mex.get('gt_damaged_share', float('nan')):.1%} of its buildings are damaged, so the damage classes are very sparse. Palu tsunami damaged/undamaged building F1 {palu.get('building_damaged_f1', float('nan')):.2f}.
- **Random raw Maxar tiles (Antakya, Kahramanmaraş, Gaziantep, İslahiye/Nurdağı):** qualitative, no ground truth. Two findings: (1) snow and cloud in the February images raise false damage in both models; (2) on clean urban tiles the two models disagree a lot (the KATE model flags ~0.3 % of pixels, the xBD model ~10 % of buildings). The KATE model flags almost nothing even in heavily damaged Antakya, so its F1 on the labelled test split does not carry over to new scenes (every KATE-CD training tile contains damage and the model runs with a high threshold).

## Data and ground truth
| set | platform | resolution | ground truth |
|---|---|---|---|
| KATE-CD (HF `CSCRS/kate-cd`) | satellite: Maxar Open Data + Airbus Pleiades | 0.3–0.5 m | hand-drawn damaged-building polygons (486 pairs, 7 provinces) |
| xBD: mexico-earthquake, palu-tsunami, sunda-tsunami | satellite: Maxar | ~0.5 m (~1 m in the model) | building polygons + 4 damage levels |
| Maxar Open Data `Kahramanmaras-turkey-earthquake-23` | satellite: Maxar (WorldView/GeoEye) | ~0.3–0.5 m | none (qualitative test) |

> Google Earth imagery cannot be downloaded programmatically under its terms of use; much of the post-disaster very-high-resolution imagery shown in Google Earth comes from Maxar. Maxar Open Data of the same class is used here.

## Results — 6 February (KATE-CD test, 38 tiles, pixel level)
{table(["method", "precision", "recall", "F1", "IoU"], [
    ["Image difference (Lab + histogram matching)", fb['baseline_test']['precision'], fb['baseline_test']['recall'], fb['baseline_test']['f1'], fb['baseline_test']['iou']],
    ["U-Net (binary), xBD only (4 shards) → zero-shot", k2['test']['precision'], k2['test']['recall'], k2['test']['f1'], k2['test']['iou']],
    ["U-Net 5-class, full xBD → zero-shot, 512 scale", kz['scale_512']['precision'], kz['scale_512']['recall'], kz['scale_512']['f1'], kz['scale_512']['iou']],
    ["U-Net 5-class, full xBD → zero-shot, 256 scale (resolution matched)", kz['scale_256']['precision'], kz['scale_256']['recall'], kz['scale_256']['f1'], kz['scale_256']['iou']],
    ["U-Net, KATE-CD only (404 training tiles)", k1['test']['precision'], k1['test']['recall'], k1['test']['f1'], k1['test']['iou']],
    ["U-Net, xBD pretraining + KATE-CD fine-tuning", k3['test']['precision'], k3['test']['recall'], k3['test']['f1'], k3['test']['iou']],
])}
{img(F / "03_predictions.png", "eq_kate_predictions.png", "KATE-CD test: pre, post, label and three models")}
{img(O / "xbd/kate_zero_shot.png", "eq_kate_zero_shot.png", "The xBD-trained 5-class model applied directly to 6 February")}

## Results — earthquake / tsunami events in xBD
{XBD_NOTE}
{xbd_rows(["earthquake", "tsunami"])}
{img(O / "xbd/samples_earthquake.png", "eq_xbd_samples.png", "Random xBD Mexico-earthquake test tiles")}

## Random cases — Maxar Open Data, 6 February (Google-Earth-class)
Random 256 m × 256 m tiles (0.5 m/px) per city, with the latest pre-event and the first post-event Maxar acquisition. Tiles are split with WorldCover into *clean urban* (built-up ≥40 %, snow/cloud <10 %) and *other* (rural, snowy or cloudy — some February 2023 images have snow and cloud).
{table(["city", "clean urban tiles", "KATE model: damaged pixels", "xBD model: major+destroyed building share", "other tiles", "KATE (other)", "xBD (other)"],
       [[c, v['clean_urban']['tiles'], v['clean_urban']['kate_damage_frac_mean'] if v['clean_urban']['tiles'] else '–', v['clean_urban']['xbd_damaged_share_mean'] if v['clean_urban']['tiles'] else '–',
         v['other']['tiles'], v['other']['kate_damage_frac_mean'] if v['other']['tiles'] else '–', v['other']['xbd_damaged_share_mean'] if v['other']['tiles'] else '–'] for c, v in mx.items() if not c.startswith('_')])}
**Failure mode:** on the {mx['_all']['n_snow_cloud']} tiles with ≥10 % snow/cloud the mean damage estimate is {mx['_all']['kate_on_snow_cloud']:.3f} (KATE model) and {mx['_all']['xbd_on_snow_cloud']:.2f} (xBD model), against {mx['_all']['kate_on_clean']:.3f} and {mx['_all']['xbd_on_clean']:.2f} on the {mx['_all']['n_clean']} clean urban tiles. Snow inflates the pre/post difference and is read as damage, so operational use needs a snow/cloud mask.
{img(O / "earthquake_maxar/city_summary.png", "eq_maxar_cities.png", "Mean prediction per city")}
{img(O / "earthquake_maxar/samples.png", "eq_maxar_samples.png", "Two clean urban tiles per city with the highest KATE prediction, plus two snowy/cloudy tiles")}

## Limitations
- The KATE-CD test split has 38 tiles and one seed, so numbers can move by a few points. Every KATE-CD tile contains damage, so the false-alarm rate is only partly measurable.
- xBD images were downsampled 2× (~1 m/px); building and damage scores would be higher at native resolution.
- The random Maxar tiles have no ground truth, and pre/post images differ in viewing angle and season — false alarms must be checked by eye.

## Sources
- KATE-CD: https://huggingface.co/datasets/CSCRS/kate-cd (ITÜ CSCRS)
{SRC_XBD}- Maxar Open Data Program: https://maxar-opendata.s3.amazonaws.com/events/catalog.json
"""

# ---------------- FLOOD ----------------
fl = J(O / "flood/results.json"); va = J(O / "flood_valencia/results.json")
fr = fl["results"]
meths = ["NDWI", "MNDWI", "S1 VV threshold", "U-Net S1", "U-Net S2", "U-Net S1+S2"]
bestv = max(va["results"], key=lambda k: va["results"][k]["f1"])
s1v = next(k for k in va["results"] if k.startswith("S1 VV threshold"))
reports["02-flood"] = f"""# Flood — Sen1Floods11 (11 countries), Valencia DANA 2024 and xBD

## Summary
- **Satellite (Sentinel-2 optical + Sentinel-1 radar, 10 m), 11 countries:** best model **U-Net S1+S2, IoU {fr['U-Net S1+S2|test']['all']['iou']:.2f}** (F1 {fr['U-Net S1+S2|test']['all']['f1']:.2f}); even the untrained NDWI threshold reaches IoU {fr['NDWI|test']['all']['iou']:.2f}. On the Bolivia event, never seen in training, U-Net S1+S2 gets IoU {fr['U-Net S1+S2|bolivia']['all']['iou']:.2f}.
- **A recent large flood — Valencia, 29 October 2024:** against the official Copernicus EMS flood extent (≈{va['gt_flood_km2']:.0f} km² in the study area) the best method is **{bestv}: F1 {va['results'][bestv]['f1']:.2f}**. Index thresholds are very precise (precision ≈0.9+) but find only about a third of the flood.
- **Urban flooding is hard from orbit:** in Valencia, recall is {va['results'][bestv]['recall_urban']:.2f} on built-up land and {va['results'][bestv]['recall_cropland']:.2f} on cropland.
- **Radar alone:** the Sentinel-1 pass two days later sees water already receding, recall {va['results'][s1v]['recall']:.2f}; timing matters.
- **Building damage (xBD flood/hurricane events):** flood-driven building damage is among the hardest types — details below.

## Data and ground truth
| set | platform | resolution | ground truth |
|---|---|---|---|
| Sen1Floods11 (HF `blumenstiel/Sen1Floods11`) | satellite: Sentinel-1 GRD + Sentinel-2 L1C | 10 m | 446 hand-labelled 512×512 chips, 11 events |
| Valencia DANA, October 2024 | satellite: Sentinel-2 L2A (31 Oct), Sentinel-1 RTC (25 Oct, 1 Nov) — Planetary Computer | 10 m | Copernicus EMS EMSR773 AOI01 flood delineation |
| xBD: midwest-flooding, nepal-flooding, 4 hurricanes | satellite: Maxar | ~0.5 m | building damage levels |

## Results — Sen1Floods11 test (90 chips) and Bolivia (unseen event, 15 chips)
Thresholds were picked on the validation split (NDWI>{fl['thresholds']['NDWI']:.2f}, MNDWI>{fl['thresholds']['MNDWI']:.2f}, VV<{fl['thresholds']['S1 VV threshold']:.0f} dB). "Flood only" excludes JRC permanent-water pixels.
{table(["method", "test IoU", "test F1", "test flood-only IoU", "Bolivia IoU", "Bolivia F1"], [[m, fr[f'{m}|test']['all']['iou'], fr[f'{m}|test']['all']['f1'], fr[f'{m}|test']['flood_only']['iou'], fr[f'{m}|bolivia']['all']['iou'], fr[f'{m}|bolivia']['all']['f1']] for m in meths])}
{img(O / "flood/iou_by_method.png", "flood_iou_by_method.png", "Method comparison")}
{img(O / "flood/iou_by_country.png", "flood_iou_by_country.png", "IoU per country/event")}
{img(O / "flood/samples.png", "flood_samples.png", "Random test chips")}
{img(O / "flood/training_curves.png", "flood_training.png", "U-Net training curves")}

## Case study — Valencia DANA (29 October 2024)
Models were transferred from Sen1Floods11 **without retraining** (the S2 model was trained on L1C and applied to L2A; band B10 was filled with its training mean). Permanent water (ESA WorldCover) is excluded.
{table(["method", "precision", "recall", "F1", "IoU", "recall (built-up)", "recall (cropland)"], [[k, v['precision'], v['recall'], v['f1'], v['iou'], v['recall_urban'], v['recall_cropland']] for k, v in va['results'].items()])}
{img(O / "flood_valencia/metrics.png", "flood_valencia_metrics.png", "Valencia: methods and recall by land cover")}
{img(O / "flood_valencia/maps.png", "flood_valencia_maps.png", "Valencia: imagery, EMS ground truth and the best prediction")}

## Results — flood/hurricane building damage (xBD)
{XBD_NOTE}
{xbd_rows(["flood", "hurricane"])}
{img(O / "xbd/samples_flood.png", "flood_xbd_samples.png", "xBD flood events — random test tiles")}

## Limitations
- The EMS ground truth was produced from the 30–31 October Landsat-8/Sentinel-2 images; methods using the same Sentinel-2 scene have an advantage, Sentinel-1 (1 Nov) a disadvantage (water had receded).
- The EMS "observed event" layer includes mud and flood traces that open-water indices cannot see, so part of the low recall is a definition gap.
- Sen1Floods11 chips are 10 m; narrow urban street flooding is invisible at that resolution.

## Sources
- Sen1Floods11 (Bonafilia et al., 2020): https://github.com/cloudtostreet/Sen1Floods11
- Copernicus EMS EMSR773: https://rapidmapping.emergency.copernicus.eu/EMSR773
- Sentinel-1/2, ESA WorldCover: Microsoft Planetary Computer
{SRC_XBD}"""

# ---------------- LANDSLIDE ----------------
ls = J(O / "landslide/results.json"); lu = J(O / "landslide_uav/results.json")
lsr, lur = ls["results"], lu["results"]
bl = max(lsr, key=lambda k: lsr[k]["f1"]); bu = max(lur, key=lambda k: lur[k]["f1"])
reports["03-landslide"] = f"""# Landslide — Landslide4Sense (satellite) and a UAV landslide set (aerial)

## Summary
- **Satellite (Sentinel-2, 14 bands: 12 spectral + slope + DEM, 10 m), Landslide4Sense test (800 chips):** best **{bl}: F1 {lsr[bl]['f1']:.2f}**, IoU {lsr[bl]['iou']:.2f}. A model using RGB only (Google-Earth-like information) reaches F1 {lsr['U-Net RGB (Google-Earth-like)']['f1']:.2f}, so the gain from multispectral bands + topography is measurable.
- **The NDVI + slope rule** gets F1 {lsr['NDVI + slope rule']['f1']:.2f}: not enough on its own.
- **Aerial (UAV, RGB, cm–dm resolution), {lu['n']['test']} test chips:** **{bu}: F1 {lur[bu]['f1']:.2f}**, IoU {lur[bu]['iou']:.2f}; the RGB soil-index rule gets F1 {lur['RGB soil index (rule)']['f1']:.2f}.
- Landslide pixels are a minority ({ls['landslide_px_frac']['test']:.1%} in the satellite set, {lu['landslide_px_frac']:.1%} in the UAV set), so the precision/recall balance is very threshold-sensitive.
- These two sets are single-date segmentation, not pre/post change detection; they show what the imagery can resolve, not the #30 pipeline itself.

## Data and ground truth
| set | platform | resolution | ground truth |
|---|---|---|---|
| Landslide4Sense (HF `ibm-nasa-geospatial/Landslide4sense`) | satellite: Sentinel-2 + ALOS PALSAR slope/DEM | ~10 m, 128×128 | pixel masks; {ls['split_sizes']['train']}/{ls['split_sizes']['validation']}/{ls['split_sizes']['test']} train/val/test |
| UAV landslide set (HF `syeddhasnainn/landslide-uav-all`, subset) | aerial: UAV RGB | cm–dm (downsampled to 384×384) | pixel masks; {lu['n']['train']}/{lu['n']['val']}/{lu['n']['test']} used |

## Results — satellite (Landslide4Sense)
Rule thresholds were picked on validation: NDVI < {ls['rule']['ndvi_lt']:.2f} and slope > {ls['rule']['slope_gt']:.2f} (normalised).
{table(["method", "precision", "recall", "F1", "IoU"], [[k, v['precision'], v['recall'], v['f1'], v['iou']] for k, v in lsr.items()])}
{img(O / "landslide/metrics.png", "landslide_l4s_metrics.png", "Landslide4Sense: metrics and precision–recall curves")}
{img(O / "landslide/samples.png", "landslide_l4s_samples.png", "Random test chips (RGB, label, rule, U-Net RGB, multispectral U-Net)")}

## Results — aerial (UAV)
{table(["method", "precision", "recall", "F1", "IoU"], [[k, v['precision'], v['recall'], v['f1'], v['iou']] for k, v in lur.items()])}
{img(O / "landslide_uav/metrics.png", "landslide_uav_metrics.png", "UAV landslides: metrics, IoU per chip, training curve")}
{img(O / "landslide_uav/samples.png", "landslide_uav_samples.png", "Random UAV test chips")}

## Limitations
- Only a subset of the UAV set was used (4 of 19 training shards); images were downsampled to 384 px.
- Landslide4Sense test regions resemble the training geography; transfer to a new region (e.g. landslides triggered by 6 February) needs its own test.

## Sources
- Landslide4Sense (Ghorbanzadeh et al., 2022): https://github.com/iarai/Landslide4Sense-2022
- UAV landslide set: https://huggingface.co/datasets/syeddhasnainn/landslide-uav-all
"""

# ---------------- TORNADO ----------------
rf = J(O / "tornado_rollingfork/results.json")
tt = xe["per_type"].get("tornado", {})
reports["04-tornado"] = f"""# Tornado — xBD (Joplin, Moore, Tuscaloosa) and Rolling Fork 2023 (aerial photography)

## Summary
- **Satellite (Maxar, xBD), 3 tornado events:** building localisation F1 {tt.get('loc_f1', float('nan')):.2f}, damaged/undamaged building F1 **{tt.get('building_damaged_f1', float('nan')):.2f}**, destroyed-building recall {tt.get('destroyed_recall', float('nan')):.2f}.
- **Aerial photography (NAIP aircraft imagery, 2021 pre / August 2023 post), Rolling Fork EF4 tornado (24 March 2023):** the xBD satellite model was applied without any training; discrimination is weak but points the right way (the mean score rises from EF0 to EF4). Against {rf['n_damage_points']} NWS survey points on structures: **EF3+ vs EF0–1 AUC {rf['auc_EF3plus_vs_EF0_1']:.2f}**, EF2+ vs undamaged control buildings AUC {rf['auc_EF2plus_vs_control']:.2f}; Spearman ρ={rf['spearman_ef_vs_score']:.2f} between EF rating and model score.

## Data and ground truth
| set | platform | resolution | ground truth |
|---|---|---|---|
| xBD tier3: joplin, moore, tuscaloosa | satellite: Maxar | ~0.5 m (~1 m in the model) | building damage levels |
| NAIP Mississippi 2021-11 and 2023-08 (Planetary Computer) | aerial: aircraft (USDA NAIP) | 0.3 m / 0.6 m → resampled to 1 m | NOAA NWS Damage Assessment Toolkit: EF-rated structure points |

## Results — xBD tornado events
{XBD_NOTE}
{xbd_rows(["tornado"])}
{img(O / "xbd/samples_tornado.png", "tornado_xbd_samples.png", "xBD tornado events — random test tiles")}

## Case study — Rolling Fork, Mississippi (EF4, 24 March 2023), aerial photography
For each NWS damage point a 256 m × 256 m pre/post NAIP chip was cut; score = the model's *major + destroyed* probability within 20 m of the point (weighted by building pixels). Controls: random built-up points (ESA WorldCover) more than 1.5 km from any damage point where the model finds a building ({rf['n_controls']} points).
{table(["level", "points", "mean score"], [[k, rf['n_by_level'][k], rf['mean_score_by_level'][k]] for k in rf['mean_score_by_level']])}
{img(O / "tornado_rollingfork/metrics.png", "tornado_rf_metrics.png", "Rolling Fork: model score by EF rating and AUC")}
{img(O / "tornado_rollingfork/samples.png", "tornado_rf_samples.png", "Random NAIP chips (cyan circle = 20 m scoring radius)")}

## Limitations
- The NAIP post image was flown ~4.5 months after the event: debris was cleared and some structures repaired, so damage looks smaller.
- NAIP (aircraft, nadir) and Maxar (satellite, oblique) differ in colour and scale; the model never saw aerial imagery.
- NWS points are damaged structures only; the "undamaged" class is approximated with control points.
- **Observed failure mode:** at EF4 points the debris had been removed by August 2023; the model reads the empty lot as "no building" and the damage score stays low (first and third rows of the sample figure). In xBD destroyed buildings always appear as rubble, so the model never learned "missing building" as damage. Scoring at the building location found in the pre image would reduce this.

## Sources
{SRC_XBD}- NAIP: Microsoft Planetary Computer `naip`
- NWS Damage Assessment Toolkit: https://apps.dat.noaa.gov/stormdamage/damageviewer/
"""

# ---------------- HAIL ----------------
hc = {c: J(O / f"hail/{c}/results.json") for c in ["A_single_event_14jun", "C_early_6to14jun", "B_season_6to19jun"]}
cname = {"A_single_event_14jun": "A: single event (14 Jun), pre 11–14 / post 16–19 Jun",
         "C_early_6to14jun": "C: events 6–14 Jun, pre 1–5 / post 11–14 Jun",
         "B_season_6to19jun": "B: events 6–19 Jun, pre 1–5 / post 16–19 Jun"}
reports["05-hail"] = f"""# Hail — Nebraska/Iowa, June 2022 (Sentinel-2 vs radar MESH)

## Summary
- Hail itself is not visible; the satellite only sees **damage to vegetation** (an NDVI drop). Ground truth is NOAA MRMS **MESH** (radar-estimated maximum hail size, ~1 km); its location was checked against SPC hail reports (median MESH at the reports ~36 mm).
- **Small/moderate hail (≥25 mm) is barely separable:** AUC {min(v['auc'] for v in hc.values()):.2f}–{max(v['auc'] for v in hc.values()):.2f}.
- **Large hail is visible:** up to AUC **{max(v['auc_by_hail_size'].get('>=60mm', 0) for v in hc.values()):.2f}** for ≥60 mm; ΔNDVI rises steadily with hail size.
- The result is very sensitive to the date window: crops grow fast in June, so the longer the gap between pre and post, the weaker the signal.

## Data and ground truth
| data | platform | resolution |
|---|---|---|
| Sentinel-2 L2A (B04, B08, SCL cloud mask) — Planetary Computer | satellite | 10 m → averaged to 0.0025° (~250 m) cells |
| ESA WorldCover 2021 (cropland mask) | satellite product | 10 m |
| NOAA MRMS MESH_Max_1440min (IEM archive) | ground radar | 0.01° (~1 km) |
| NOAA SPC hail reports | observers | points |

## Results
Positive: MESH ≥ 25 mm; negative: MESH < 10 mm over the same period. Score: ΔNDVI = NDVI(pre) − NDVI(post), cropland cells only.
{table(["configuration", "valid cells", "hail cells", "AUC ≥25 mm", "AUC ≥40 mm", "AUC ≥60 mm", "Spearman (MESH, ΔNDVI)", "best F1*"],
       [[cname[c], v['valid_cells'], v['hail_cells'], v['auc'], v['auc_by_hail_size'].get('>=40mm', float('nan')), v['auc_by_hail_size'].get('>=60mm', float('nan')), v['spearman_mesh_vs_dndvi'], v['best_f1']] for c, v in hc.items()])}
\\* The F1 threshold was picked on the same data (optimistic); the threshold-free AUC is the real comparison.

{img(O / "hail/C_early_6to14jun/metrics.png", "hail_C_metrics.png", "Configuration C: ROC, ΔNDVI by MESH class, threshold curves")}
{img(O / "hail/C_early_6to14jun/maps.png", "hail_C_maps.png", "Configuration C: NDVI pre/post, MESH and ΔNDVI maps")}
{img(O / "hail/A_single_event_14jun/maps.png", "hail_A_maps.png", "Configuration A: the 14 June event")}

## Limitations
- MESH is a radar estimate (it often overestimates hail size and does not measure hail at the ground) — the best available area measurement, not truth.
- ~250 m cells mix field boundaries; field-level analysis and a longer time series would strengthen the signal.
- In early June corn and soy are small, so damage shows little in NDVI; July–August events may be clearer.

## Sources
- NOAA MRMS (IEM archive): https://mtarchive.geol.iastate.edu/
- NOAA SPC Storm Reports: https://www.spc.noaa.gov/climo/reports/
- CIMSS satellite blog, Nebraska/Iowa hail swaths (June 2022): https://cimss.ssec.wisc.edu/satellite-blog/archives/46975
"""

# ---------------- EXTREME HEAT ----------------
he = J(O / "heat/results.json"); pc = he["per_city"]
reports["06-extreme-heat"] = f"""# Extreme heat — summer 2023 heatwaves (MODIS surface temperature vs stations)

## Summary
- **10 cities** (Phoenix, Las Vegas, El Paso, Seville, Rome, Palermo, Athens, Antalya, Adana, Beijing), June–August 2023; calibration uses summer 2022 only.
- Daytime satellite land-surface temperature (LST) vs station daily maximum air temperature: mean **r = {he['mean_r']:.2f}**; linear calibration error **{he['mean_rmse']:.1f} °C** RMSE on average.
- **Extreme-heat day detection** (station Tmax ≥ 2013–2022 summer P90): ranking is good (**pooled AUC {he['pooled_auc_zscore']:.2f}**), but the thresholding rule decides the result: linear calibration F1 {he['pooled_f1']:.2f} (estimates shrink towards the mean), quantile matching F1 **{he['pooled_q_f1']:.2f}** (precision {he['pooled_q_precision']:.2f}, recall {he['pooled_q_recall']:.2f}).
- The Mediterranean LST anomaly map clearly shows the July 2023 "Cerberus" heatwave over Sicily, Sardinia, Greece, North Africa and around Adana (land mean {he.get('anomaly_mean_land_c', float('nan')):+.1f} °C).
- This is a single-date measurement, not pre/post damage detection; it is outside the #30 change-detection pipeline.

## Data and ground truth
| data | platform | resolution |
|---|---|---|
| MODIS Terra MOD11A1 daily daytime LST — Planetary Computer | satellite | 1 km |
| MODIS Terra MOD11A2 8-day LST (anomaly map) | satellite | 1 km → 0.05° |
| Meteostat (NOAA ISD/DWD stations) daily Tmax, 2013–2023 | ground station | point |

## Results — per city (test year 2023)
{table(["city", "station", "days (clear)", "extreme days", "r", "RMSE °C", "AUC", "F1 linear", "F1 quantile"],
       [[c, v['station'], v['n_days'], v['heat_days'], v['pearson_r'], v['rmse_c'], v['auc'] if v['auc'] is not None else 'n/a', v['f1'], v['q_f1']] for c, v in pc.items()])}
{img(O / "heat/metrics.png", "heat_metrics.png", "LST–Tmax relation, extreme-day ROC and per-city performance")}
{img(O / "heat/timeseries.png", "heat_timeseries.png", "Summer 2023 time series: station Tmax and satellite estimate")}
{img(O / "heat/anomaly_map.png", "heat_anomaly_map.png", "12–19 July 2023 LST anomaly (vs the 2018–2022 mean)")}

## Limitations
- LST is surface, not air temperature; urban surfaces and bare soil can be 10–20 °C hotter during the day.
- No measurement on cloudy days (see the days column); in Beijing the monsoon removes about half of the summer days.
- 2023 was clearly hotter than 2022: thresholds calibrated on the 2022 rate under-predict the number of extreme days in 2023 (low recall).

## Sources
- MODIS LST (Wan et al.), Microsoft Planetary Computer `modis-11A1-061`, `modis-11A2-061`
- Meteostat: https://meteostat.net
"""

# ---------------- FIRE ----------------
fi = J(O / "fire/results.json")
fx = xe["per_type"].get("fire", {})
reports["07-fire"] = f"""# Fire — burned area (Sentinel-2 dNBR) and building damage (xBD)

## Summary
- **Burned area, 4 fires, against official perimeters:** {', '.join(f"{n.split(' (')[0]} IoU {max(v['iou'] for v in r['scores'].values()):.2f}" for n, r in fi.items())} (best threshold).
- A single standard threshold (USGS dNBR > 0.10) works well without training on most fires; errors come from unburned islands inside the official perimeter, harvested fields and cloud.
- **Fire-driven building damage (xBD, 5 fire events):** damaged/undamaged building F1 **{fx.get('building_damaged_f1', float('nan')):.2f}**, destroyed-building recall {fx.get('destroyed_recall', float('nan')):.2f} — fire is one of the best-detected hazards in xBD (burned buildings are binary: standing or ash).

## Data and ground truth
| data | platform | resolution | ground truth |
|---|---|---|---|
| Sentinel-2 L2A B08/B12 pre/post mosaics — Planetary Computer | satellite | 20 m (~25 m grid) | NIFC WFIGS / InterAgency fire perimeters (US), EFFIS burned area (Manavgat) |
| xBD: socal, santa-rosa, woolsey, portugal, pinery | satellite: Maxar | ~0.5 m (~1 m in the model) | building damage levels |

## Results — burned area
{table(["fire", "method", "precision", "recall", "F1", "IoU", "official area km²", "satellite area km²**"],
       [[n, k, v['precision'], v['recall'], v['f1'], v['iou'], v['area_official_km2'], v['area_pred_km2']] for n, r in fi.items() for k, v in r['scores'].items()])}
\\*\\* The satellite area is measured over the whole perimeter box with a 25 % buffer (other burns inside the box count too).

{img(O / "fire/metrics.png", "fire_metrics.png", "IoU, area comparison and burn-severity mix inside the perimeter")}
{img(O / "fire/maps.png", "fire_maps.png", "NBR pre/post, dNBR and error map")}

## Results — building damage (xBD fire events)
{XBD_NOTE}
{xbd_rows(["fire"])}
{img(O / "xbd/samples_fire.png", "fire_xbd_samples.png", "xBD fire events — random test tiles")}

## Limitations
- Official perimeters include unburned islands, so the IoU ceiling is below 1.
- The EFFIS perimeter is MODIS/VIIRS + Sentinel-2 based, the NIFC perimeters are airborne/field mapped — consistency differs between sources.
- For the Camp fire the December "post" mosaic has cloud/snow and harvested fields that produce false alarms.

## Sources
- NIFC WFIGS Interagency Perimeters: https://data-nifc.opendata.arcgis.com
- EFFIS: https://effis.jrc.ec.europa.eu
{SRC_XBD}"""

for old in R.glob("0*.md"):
    old.unlink()
for name, text in reports.items():
    (R / f"{name}.md").write_text(text)

heads = [
    ("Earthquake\n(6 Feb, KATE-CD)", k1["test"]["f1"], "F1 (pixel)"),
    ("Flood\n(Sen1Floods11)", fr["U-Net S1+S2|test"]["all"]["f1"], "F1 (pixel)"),
    ("Flood\n(Valencia 2024)", va["results"][bestv]["f1"], "F1 (pixel)"),
    ("Landslide\n(satellite)", lsr[bl]["f1"], "F1 (pixel)"),
    ("Landslide\n(UAV)", lur[bu]["f1"], "F1 (pixel)"),
    ("Tornado\n(xBD buildings)", tt.get("building_damaged_f1", 0), "F1 (building)"),
    ("Tornado\n(NAIP, EF3+ vs EF0–1)", rf["auc_EF3plus_vs_EF0_1"], "AUC"),
    ("Hail\n(≥60 mm)", max(v["auc_by_hail_size"].get(">=60mm", 0) for v in hc.values()), "AUC"),
    ("Hail\n(≥25 mm)", max(v["auc"] for v in hc.values()), "AUC"),
    ("Extreme heat\n(day detection)", he["pooled_auc_zscore"], "AUC"),
    ("Fire\n(burned area)", float(np.mean([max(v['f1'] for v in r['scores'].values()) for r in fi.values()])), "F1 (pixel)"),
    ("Fire\n(xBD buildings)", fx.get("building_damaged_f1", 0), "F1 (building)"),
]
fig, ax = plt.subplots(figsize=(15, 5))
cols = {"F1 (pixel)": "#3b6ea5", "F1 (building)": "#6a994e", "AUC": "#e07b39"}
for i, (n, v, m) in enumerate(heads):
    ax.bar(i, v, color=cols[m]); ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(range(len(heads)), [h[0] for h in heads], fontsize=8); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.3)
ax.legend(handles=[Patch(color=c, label=l) for l, c in cols.items()], loc="upper right")
ax.set_title("Headline metric of the best method per hazard (different metrics are not directly comparable)")
plt.tight_layout(); plt.savefig(IMG / "overview.png", dpi=100); plt.close()

index = """# DamageLens — satellite and aerial image analysis by hazard type

Each report: data and ground truth, methods, metric tables, charts, random-case figures, limitations and sources.
Code and raw outputs: `experiments/multi-hazard/` (one script per experiment, results in `experiments/multi-hazard/outputs/`).

![Overview](img/overview.png)

| hazard | report | platforms | headline result |
|---|---|---|---|
"""
rows = [("Earthquake", "01-earthquake.md", "VHR satellite (Maxar/Pleiades — Google-Earth-class)", f"6 February F1 {k1['test']['f1']:.2f}; transfer from xBD F1 {best_kz['f1']:.2f}"),
        ("Flood", "02-flood.md", "satellite (Sentinel-1/2), VHR satellite (xBD)", f"11 countries IoU {fr['U-Net S1+S2|test']['all']['iou']:.2f}; Valencia 2024 F1 {va['results'][bestv]['f1']:.2f}"),
        ("Landslide", "03-landslide.md", "satellite (Sentinel-2 + DEM), aerial (UAV)", f"satellite F1 {lsr[bl]['f1']:.2f}; UAV F1 {lur[bu]['f1']:.2f}"),
        ("Tornado", "04-tornado.md", "VHR satellite (xBD), aerial (NAIP aircraft)", f"xBD building F1 {tt.get('building_damaged_f1', 0):.2f}; Rolling Fork EF3+ AUC {rf['auc_EF3plus_vs_EF0_1']:.2f}"),
        ("Hail", "05-hail.md", "satellite (Sentinel-2) + radar MESH", f"≥60 mm AUC {max(v['auc_by_hail_size'].get('>=60mm', 0) for v in hc.values()):.2f}; ≥25 mm AUC {max(v['auc'] for v in hc.values()):.2f}"),
        ("Extreme heat", "06-extreme-heat.md", "satellite (MODIS LST) + stations", f"r {he['mean_r']:.2f}; day detection AUC {he['pooled_auc_zscore']:.2f}"),
        ("Fire", "07-fire.md", "satellite (Sentinel-2), VHR satellite (xBD)", f"burned area IoU {min(max(v['iou'] for v in r['scores'].values()) for r in fi.values()):.2f}–{max(max(v['iou'] for v in r['scores'].values()) for r in fi.values()):.2f}; xBD building F1 {fx.get('building_damaged_f1', 0):.2f}")]
for a, f, p, s in rows:
    index += f"| {a} | [{f}]({f}) | {p} | {s} |\n"
index += """
**About Google Earth:** Google Earth imagery cannot be downloaded in bulk or programmatically under its terms of use. Much of the post-disaster very-high-resolution imagery in Google Earth comes from Maxar; this work uses Maxar Open Data of the same class (xBD, KATE-CD, the Kahramanmaraş 2023 event). Aerial imagery was tested with UAV (landslide) and aircraft (USDA NAIP, tornado) data; no open, labelled helicopter dataset was found.
"""
(R / "README.md").write_text(index)
print("written", list(reports))
