import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
import segmentation_models_pytorch as smp
import torch
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from eo import CAT
from seg import DEV

ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "tornado_rollingfork"
OUT.mkdir(parents=True, exist_ok=True)
BBOX = [-91.0, 32.80, -90.80, 33.00]
CHIP, GSD = 256, 1.0
UTM = "EPSG:32615"
to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)

q = requests.get("https://services.dat.noaa.gov/arcgis/rest/services/nws_damageassessmenttoolkit/DamageViewer/FeatureServer/0/query",
                 params=dict(where=f"stormdate >= DATE '2023-03-24' AND stormdate < DATE '2023-03-26' AND lat > {BBOX[1]} AND lat < {BBOX[3]} AND lon > {BBOX[0]} AND lon < {BBOX[2]}",
                             outFields="efscale,dod,damage_txt,lat,lon", returnGeometry="false", f="json", resultRecordCount=2000), timeout=120).json()
pts = [f["attributes"] for f in q["features"]]
pts = [p for p in pts if p["efscale"] in ("EF0", "EF1", "EF2", "EF3", "EF4") and not any(k in (p["damage_txt"] or "") for k in ("Trees", "Transmission", "Tower"))]
print("structure damage points", len(pts), flush=True)

items = list(CAT.search(collections=["naip"], bbox=BBOX).items())
pre_items = [i for i in items if i.datetime.year == 2021]
post_items = [i for i in items if i.datetime.year == 2023]


def chip(items_, x, y):
    tr = from_origin(x - CHIP * GSD / 2, y + CHIP * GSD / 2, GSD, GSD)
    out = np.zeros((CHIP, CHIP, 3), np.uint8)
    for it in items_:
        with rasterio.open(it.assets["image"].href) as src, WarpedVRT(src, crs=UTM, transform=tr, width=CHIP, height=CHIP, resampling=Resampling.average) as v:
            a = v.read([1, 2, 3]).transpose(1, 2, 0)
            m = (out.sum(-1) == 0) & (a.sum(-1) > 0); out[m] = a[m]
    return out


# controls: grid points in the same towns, >1.5 km from any reported damage point
lonlat = np.array([[p["lon"], p["lat"]] for p in pts]); xy = np.array(to_utm.transform(lonlat[:, 0], lonlat[:, 1])).T
rng = np.random.default_rng(0)
from eo import grid, worldcover
WRES = 0.0002
wc = worldcover(BBOX, WRES); wtr, WW, WH = grid(BBOX, WRES)
rr, cc = np.where(wc == 50)  # built-up pixels only
ctrl = []
for k in rng.permutation(len(rr)):
    lo, la = BBOX[0] + (cc[k] + .5) * WRES, BBOX[3] - (rr[k] + .5) * WRES
    cx, cy = to_utm.transform(lo, la)
    if np.min(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)) > 1500:
        ctrl.append(dict(lon=float(lo), lat=float(la), efscale="none"))
    if len(ctrl) >= 300:
        break
print("controls on built-up land", len(ctrl), flush=True)

model = smp.Unet("resnet18", encoder_weights=None, in_channels=6, classes=5).to(DEV)
model.load_state_dict(torch.load(ROOT / "outputs/xbd/xbd_unet5.pt", map_location=DEV)); model.eval()
mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=DEV).view(1, 6, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225] * 2, device=DEV).view(1, 6, 1, 1)
yy, xx = np.mgrid[:CHIP, :CHIP]; disk = np.hypot(yy - CHIP / 2, xx - CHIP / 2) <= 20


def fetch(p):
    x, y = to_utm.transform(p["lon"], p["lat"])
    return p, chip(pre_items, x, y), chip(post_items, x, y)


def score(p, a, b):
    t = (torch.from_numpy(np.concatenate([a, b], -1)[None]).to(DEV).permute(0, 3, 1, 2).float() / 255 - mean) / std
    with torch.no_grad():
        pr = model(t).softmax(1)[0].cpu().numpy()
    bldg = pr[1:].sum(0)
    w = disk * bldg
    dmg = float((pr[3:].sum(0) * disk).sum() / max(w.sum(), 1e-6)) if w.sum() > 5 else float(pr[3:][:, disk].sum(0).mean())
    return dict(**p, dmg_score=dmg, building_px=float(w.sum()), pred_class=int(pr.argmax(0)[disk].max())), (a, b, pr.argmax(0))


cache = OUT / "scores.json"
with ThreadPoolExecutor(8) as ex:
    fetched = list(ex.map(fetch, pts + ctrl))
print("chips fetched", flush=True)
rows, chips = [], []
for k, (p, a, b) in enumerate(fetched):
    r, c = score(p, a, b); rows.append(r)
    if len(chips) < 6 and p["efscale"] in ("EF3", "EF4", "none") and rng.random() < .3:
        chips.append((p["efscale"], c))
    if k % 100 == 0:
        print(k, len(fetched), flush=True)
json.dump(rows, open(cache, "w"))
rows = [r for r in rows if r["efscale"] != "none" or r["building_px"] > 10]  # controls must sit on a building

ef = np.array([r["efscale"] for r in rows]); s = np.array([r["dmg_score"] for r in rows])
lvl = {"none": -1, "EF0": 0, "EF1": 1, "EF2": 2, "EF3": 3, "EF4": 4}
L = np.array([lvl[e] for e in ef])
res = dict(n_damage_points=int((L >= 0).sum()), n_controls=int((L < 0).sum()),
           auc_EF2plus_vs_control=float(roc_auc_score((L[(L >= 2) | (L < 0)] >= 2).astype(int), s[(L >= 2) | (L < 0)])),
           auc_EF3plus_vs_EF0_1=float(roc_auc_score((L[(L >= 3) | ((L >= 0) & (L <= 1))] >= 3).astype(int), s[(L >= 3) | ((L >= 0) & (L <= 1))])),
           auc_any_vs_control=float(roc_auc_score((L >= 0).astype(int), s)),
           spearman_ef_vs_score=float(spearmanr(L[L >= 0], s[L >= 0]).statistic),
           mean_score_by_level={k: float(s[L == v].mean()) for k, v in lvl.items() if (L == v).any()},
           n_by_level={k: int((L == v).sum()) for k, v in lvl.items()},
           imagery=dict(pre=[i.id for i in pre_items], post=[i.id for i in post_items]))
json.dump(res, open(OUT / "results.json", "w"), indent=2)
print(json.dumps(res, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
keys = [k for k in lvl if (L == lvl[k]).any()]
ax[0].boxplot([s[L == lvl[k]] for k in keys], tick_labels=["no damage\n(control)" if k == "none" else k for k in keys], showfliers=False)
ax[0].set_ylabel("model P(major + destroyed) within 20 m"); ax[0].grid(alpha=.3)
ax[0].set_title(f"Rolling Fork EF4 tornado: model score by NWS EF rating (Spearman ρ={res['spearman_ef_vs_score']:.2f})")
ax[1].bar(["any damage\nvs control", "EF2+\nvs control", "EF3+\nvs EF0–1"], [res["auc_any_vs_control"], res["auc_EF2plus_vs_control"], res["auc_EF3plus_vs_EF0_1"]], color="#3b6ea5")
ax[1].axhline(.5, color="k", ls="--", lw=.7); ax[1].set_ylim(0, 1); ax[1].set_ylabel("ROC AUC"); ax[1].grid(axis="y", alpha=.3); ax[1].set_title("Discrimination")
plt.tight_layout(); plt.savefig(OUT / "metrics.png", dpi=100); plt.close()

COL = np.array([[0, 0, 0], [60, 180, 75], [255, 225, 25], [245, 130, 48], [230, 25, 75]], np.uint8)
fig, ax = plt.subplots(len(chips), 3, figsize=(10, 3.4 * len(chips)), squeeze=False)
for r, (e, (a, b, p)) in enumerate(chips):
    ax[r, 0].imshow(a); ax[r, 0].set_title(f"NAIP 2021 (pre) — {e}", fontsize=9)
    ax[r, 1].imshow(b); ax[r, 1].set_title("NAIP Aug 2023 (post)", fontsize=9)
    ax[r, 2].imshow(COL[p]); ax[r, 2].set_title("xBD model", fontsize=9)
    for k in range(3):
        ax[r, k].add_patch(plt.Circle((CHIP / 2, CHIP / 2), 20, fill=False, color="c")); ax[r, k].axis("off")
plt.tight_layout(); plt.savefig(OUT / "samples.png", dpi=75); plt.close()
