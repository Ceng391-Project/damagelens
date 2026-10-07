import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from eo import read_to_grid, s2_mosaic, worldcover

ROOT = Path(__file__).parent
D = ROOT / "data" / "hail"
OUT = ROOT / "outputs" / "hail"
OUT.mkdir(parents=True, exist_ok=True)

BBOX = [-100.2, 40.2, -95.8, 41.5]
RES = 0.0025
import sys
CONFIGS = {
    # name: (pre window, post window, MESH days counted as hail, MESH days that must stay <10 mm for negatives)
    "A_single_event_14jun": ("2022-06-11/2022-06-14", "2022-06-16/2022-06-19", ["20220615"], [f"202206{d:02d}" for d in range(12, 20)]),
    "B_season_6to19jun": ("2022-06-01/2022-06-05", "2022-06-16/2022-06-19", [f"202206{d:02d}" for d in range(6, 20)], [f"202206{d:02d}" for d in range(6, 20)]),
    "C_early_6to14jun": ("2022-06-01/2022-06-05", "2022-06-11/2022-06-14", [f"202206{d:02d}" for d in range(6, 15)], [f"202206{d:02d}" for d in range(6, 15)]),
}
CFG = sys.argv[1]
PRE, POST, EVENT_DAYS, DAYS = CONFIGS[CFG]
OUT = OUT / CFG
OUT.mkdir(parents=True, exist_ok=True)

import gzip, re, urllib.request
for d in sorted(set(DAYS + EVENT_DAYS)):
    f = D / f"mesh_{d}.grib2"
    if not f.exists():
        base = f"https://mtarchive.geol.iastate.edu/{d[:4]}/{d[4:6]}/{d[6:]}/mrms/ncep/MESH_Max_1440min/"
        names = sorted(set(re.findall(r"M[A-Z]+_Max_1440min_00\.50_\d{8}-\d{6}\.grib2\.gz", urllib.request.urlopen(base).read().decode())))
        name = min(names, key=lambda n: abs(int(n[-16:-10]) - 120000))
        url = base + name
        f.write_bytes(gzip.decompress(urllib.request.urlopen(url).read()))

rd = lambda d: np.clip(np.nan_to_num(read_to_grid(str(D / f"mesh_{d}.grib2"), BBOX, RES, Resampling.nearest), nan=0), 0, None)
mesh = np.max([rd(d) for d in EVENT_DAYS], 0)
mesh_any = np.max([rd(d) for d in DAYS], 0)

cache = OUT / f"s2_cache_{POST[:10]}.npz"
if cache.exists():
    z = np.load(cache); pre_r, pre_n, post_r, post_n = z["pre_r"], z["pre_n"], z["post_r"], z["post_n"]
    pre_ids, post_ids = list(z["pre_ids"]), list(z["post_ids"])
else:
    pre, pre_ids = s2_mosaic(BBOX, PRE, RES)
    post, post_ids = s2_mosaic(BBOX, POST, RES)
    pre_r, pre_n, post_r, post_n = pre["B04"], pre["B08"], post["B04"], post["B08"]
    np.savez_compressed(cache, pre_r=pre_r, pre_n=pre_n, post_r=post_r, post_n=post_n, pre_ids=pre_ids, post_ids=post_ids)
wc = worldcover(BBOX, RES)

ndvi = lambda r, n: (n - r) / (n + r + 1e-6)
d_raw = ndvi(pre_r, pre_n) - ndvi(post_r, post_n)  # positive = vegetation loss
crop = wc == 40
d_ndvi = d_raw
valid = np.isfinite(d_ndvi) & crop
hail = mesh >= 25
nohail = mesh_any < 10
ev = valid & (hail | nohail)
y, s = hail[ev].astype(int), d_ndvi[ev]
auc = roc_auc_score(y, s)
fpr, tpr, thr = roc_curve(y, s)
f1s = []
for t in np.linspace(-0.1, 0.4, 51):
    p = s > t; tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = (~p & (y == 1)).sum()
    f1s.append((2 * tp / max(2 * tp + fp + fn, 1), t, tp / max(tp + fp, 1), tp / max(tp + fn, 1)))
best = max(f1s)
auc_by_size = {}
for lo_mm in (25, 40, 60):
    pos = valid & (mesh >= lo_mm); ev2 = pos | (valid & nohail)
    if pos.sum() > 50:
        auc_by_size[f">={lo_mm}mm"] = float(roc_auc_score(pos[ev2].astype(int), d_ndvi[ev2]))
rho = spearmanr(mesh[valid & (mesh > 0)], d_ndvi[valid & (mesh > 0)]).statistic
bins = [(0, 0.01, "0"), (0.01, 10, "<10"), (10, 15, "10–15"), (15, 25, "15–25"), (25, 40, "25–40"), (40, 60, "40–60"), (60, 200, ">60")]
by_bin = {lab: d_ndvi[valid & (mesh >= lo) & (mesh < hi)] for lo, hi, lab in bins}

# control: same pixels' NDVI change in an untouched cropland window far from swaths
res = dict(event="Nebraska–Iowa hail swaths, June 2022", config=CFG, mesh_days=EVENT_DAYS, bbox=BBOX, pre_window=PRE, post_window=POST,
           pre_scenes=len(pre_ids), post_scenes=len(post_ids), cropland_cells=int(crop.sum()), valid_cells=int(valid.sum()),
           hail_cells=int((valid & hail).sum()), auc=float(auc), best_f1=float(best[0]), best_thr=float(best[1]),
           precision=float(best[2]), recall=float(best[3]), spearman_mesh_vs_dndvi=float(rho), auc_by_hail_size=auc_by_size,
           median_dndvi_by_mesh_bin={k: float(np.median(v)) if len(v) else None for k, v in by_bin.items()},
           n_by_mesh_bin={k: int(len(v)) for k, v in by_bin.items()})
json.dump(res, open(OUT / "results.json", "w"), indent=2)
print(json.dumps(res, indent=1))

ext = [BBOX[0], BBOX[2], BBOX[1], BBOX[3]]
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
rgb = lambda r, n: np.dstack([np.clip(r * 4, 0, 1), np.clip(n * 2, 0, 1), np.clip(r * 4, 0, 1)])
ax[0, 0].imshow(np.nan_to_num(ndvi(pre_r, pre_n)), cmap="YlGn", vmin=0, vmax=0.8, extent=ext); ax[0, 0].set_title(f"NDVI öncesi ({PRE})")
ax[0, 1].imshow(np.nan_to_num(ndvi(post_r, post_n)), cmap="YlGn", vmin=0, vmax=0.8, extent=ext); ax[0, 1].set_title(f"NDVI sonrası ({POST})")
im = ax[1, 0].imshow(np.where(mesh > 0, mesh, np.nan), cmap="magma_r", vmin=0, vmax=80, extent=ext); ax[1, 0].set_title("MESH maksimum dolu boyutu (mm) — yer gerçeği")
plt.colorbar(im, ax=ax[1, 0], fraction=.03)
im = ax[1, 1].imshow(np.where(crop, d_ndvi, np.nan), cmap="RdYlGn_r", vmin=-0.2, vmax=0.4, extent=ext)
ax[1, 1].contour(np.flipud(hail).astype(float), levels=[0.5], colors="k", linewidths=.6, extent=ext, origin="lower")
ax[1, 1].set_title("ΔNDVI (öncesi − sonrası), tarım alanı; siyah = MESH ≥25 mm"); plt.colorbar(im, ax=ax[1, 1], fraction=.03)
for a in ax.ravel(): a.set_xlabel("boylam"); a.set_ylabel("enlem")
plt.tight_layout(); plt.savefig(OUT / "maps.png", dpi=90); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(fpr, tpr, color="#3b6ea5"); ax[0].plot([0, 1], [0, 1], "k--", lw=.7)
ax[0].set_title(f"ROC: ΔNDVI ile dolu tespiti (AUC={auc:.2f})"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].grid(alpha=.3)
labs = [b[2] for b in bins]
ax[1].boxplot([by_bin[l] for l in labs], tick_labels=labs, showfliers=False)
ax[1].set_xlabel("MESH (mm)"); ax[1].set_ylabel("ΔNDVI"); ax[1].set_title(f"Dolu boyutu arttıkça bitki kaybı (Spearman ρ={rho:.2f})"); ax[1].grid(alpha=.3)
ts = [f[1] for f in f1s]
ax[2].plot(ts, [f[0] for f in f1s], label="F1"); ax[2].plot(ts, [f[2] for f in f1s], label="precision"); ax[2].plot(ts, [f[3] for f in f1s], label="recall")
ax[2].axvline(best[1], color="k", ls=":"); ax[2].set_xlabel("ΔNDVI eşiği"); ax[2].legend(); ax[2].grid(alpha=.3); ax[2].set_title("Eşik seçimi")
plt.tight_layout(); plt.savefig(OUT / "metrics.png", dpi=100); plt.close()
