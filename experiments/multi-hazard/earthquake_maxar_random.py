import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from pyproj import Transformer

from eo import worldcover

from seg import DEV

ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "earthquake_maxar"
OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(ROOT.parent.parent / "data/maxar_tr/index.json"))
CITIES = {"Antakya (Hatay)": [36.10, 36.15, 36.25, 36.40], "Kahramanmaraş centre": [36.88, 37.55, 36.97, 37.61],
          "Gaziantep centre": [37.30, 37.03, 37.43, 37.10], "İslahiye / Nurdağı": [36.55, 36.98, 36.72, 37.20]}
N_PER_CITY, SIZE, GSD = 16, 512, 0.5
rng = np.random.default_rng(2023)


def inside(b, box):
    return b[0] < box[2] and b[2] > box[0] and b[1] < box[3] and b[3] > box[1]


pairs = defaultdict(list)
byq = defaultdict(lambda: {"pre": [], "post": []})
for m in meta:
    if m["visual"]:
        byq[m["qk"]]["pre" if m["dt"] < "2023-02-06" else "post"].append(m)
for q, d in byq.items():
    if d["pre"] and d["post"]:
        for c, box in CITIES.items():
            if inside(d["pre"][0]["bbox"], box):
                pre = max(d["pre"], key=lambda m: m["dt"]); post = min(d["post"], key=lambda m: m["dt"])
                pairs[c].append((q, pre, post))
print({c: len(v) for c, v in pairs.items()}, flush=True)


def window(m, x0, y0, crs):
    tr = from_origin(x0, y0, GSD, GSD)
    with rasterio.open(m["visual"]) as src, WarpedVRT(src, crs=crs, transform=tr, width=SIZE, height=SIZE, resampling=Resampling.average) as v:
        return v.read([1, 2, 3]).transpose(1, 2, 0)


def sample(args):
    c, (q, pre, post), k = args
    r = np.random.default_rng(k * 7919 + int(q) % 1000)
    with rasterio.open(post["visual"]) as src:
        crs, b = src.crs, src.bounds
    for _ in range(25):
        x0 = r.uniform(b.left, b.right - SIZE * GSD); y0 = r.uniform(b.bottom + SIZE * GSD, b.top)
        a, p = window(pre, x0, y0, crs), window(post, x0, y0, crs)
        if (a.sum(-1) > 0).mean() < .95 or (p.sum(-1) > 0).mean() < .95 or a.std() < 15:
            continue
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x0 + SIZE * GSD / 2, y0 - SIZE * GSD / 2)
        bright = float(((p.min(-1) > 190) & (p.max(-1) - p.min(-1) < 30)).mean())  # snow / cloud
        return dict(city=c, qk=q, pre=pre["dt"][:10], post=post["dt"][:10], lon=lon, lat=lat, snow_cloud_frac=bright), a, p
    return None


jobs = []
for c, lst in pairs.items():
    for k in range(N_PER_CITY):
        jobs.append((c, lst[rng.integers(len(lst))], k))
with ThreadPoolExecutor(8) as ex:
    samples = [s for s in ex.map(sample, jobs) if s]
for info, a, p in samples:
    d = 0.0015
    w = worldcover([info["lon"] - d, info["lat"] - d, info["lon"] + d, info["lat"] + d], 0.0001)
    info["builtup_frac"] = float((w == 50).mean())
    info["clean_urban"] = info["builtup_frac"] >= 0.4 and info["snow_cloud_frac"] < 0.1
print("tiles", len(samples), "clean urban", sum(s[0]["clean_urban"] for s in samples), flush=True)

kate = smp.Unet("resnet18", encoder_weights=None, in_channels=6, classes=1).to(DEV)
kate.load_state_dict(torch.load(ROOT.parent / "feasibility/outputs/kate_only.pt", map_location=DEV)); kate.eval()
kate_thr = json.load(open(ROOT.parent / "feasibility/outputs/kate_only.json"))["thr"]
xbd = smp.Unet("resnet18", encoder_weights=None, in_channels=6, classes=5).to(DEV)
xbd.load_state_dict(torch.load(ROOT / "outputs/xbd/xbd_unet5.pt", map_location=DEV)); xbd.eval()
mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=DEV).view(1, 6, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225] * 2, device=DEV).view(1, 6, 1, 1)
rows, vis = [], []
with torch.no_grad():
    for info, a, p in samples:
        t = (torch.from_numpy(np.concatenate([a, p], -1)[None]).to(DEV).permute(0, 3, 1, 2).float() / 255 - mean) / std
        k = (torch.sigmoid(kate(t))[0, 0].cpu().numpy() > kate_thr)
        t2 = torch.nn.functional.interpolate(t, size=SIZE // 2, mode="area")  # xBD model was trained at ~1 m/px
        pr = torch.nn.functional.interpolate(xbd(t2).softmax(1), size=SIZE, mode="bilinear")[0].cpu().numpy()
        cls = pr.argmax(0); bld = cls > 0
        rows.append(dict(**info, kate_damage_frac=float(k.mean()), xbd_building_frac=float(bld.mean()),
                         xbd_damaged_share_of_buildings=float((cls >= 3).sum() / max(bld.sum(), 1))))
        vis.append((info, a, p, k, cls))
json.dump(rows, open(OUT / "tiles.json", "w"), indent=2)
def agg(sel):
    return dict(tiles=len(sel), kate_damage_frac_mean=float(np.mean([r["kate_damage_frac"] for r in sel])) if sel else None,
                xbd_damaged_share_mean=float(np.mean([r["xbd_damaged_share_of_buildings"] for r in sel])) if sel else None)
summary = {c: dict(clean_urban=agg([r for r in rows if r["city"] == c and r["clean_urban"]]),
                   other=agg([r for r in rows if r["city"] == c and not r["clean_urban"]]))
           for c in CITIES if any(r["city"] == c for r in rows)}
summary_all = dict(kate_on_snow_cloud=float(np.mean([r["kate_damage_frac"] for r in rows if r["snow_cloud_frac"] >= .1] or [np.nan])),
                   kate_on_clean=float(np.mean([r["kate_damage_frac"] for r in rows if r["clean_urban"]] or [np.nan])),
                   xbd_on_snow_cloud=float(np.mean([r["xbd_damaged_share_of_buildings"] for r in rows if r["snow_cloud_frac"] >= .1] or [np.nan])),
                   xbd_on_clean=float(np.mean([r["xbd_damaged_share_of_buildings"] for r in rows if r["clean_urban"]] or [np.nan])),
                   n_snow_cloud=sum(r["snow_cloud_frac"] >= .1 for r in rows), n_clean=sum(r["clean_urban"] for r in rows), n_total=len(rows))
summary["_all"] = summary_all
json.dump(summary, open(OUT / "results.json", "w"), indent=2)
print(json.dumps(summary, indent=1))

cs = [c for c in summary if not c.startswith("_")]
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5)); x = np.arange(len(cs))
nz = lambda v: 0 if v is None else v
ax[0].bar(x - .2, [nz(summary[c]["clean_urban"]["kate_damage_frac_mean"]) for c in cs], .4, label="KATE-CD model: damaged pixel fraction")
ax[0].bar(x + .2, [nz(summary[c]["clean_urban"]["xbd_damaged_share_mean"]) for c in cs], .4, label="xBD model: major+destroyed share of buildings")
ax[0].set_xticks(x, [f"{c}\n(n={summary[c]['clean_urban']['tiles']})" for c in cs], fontsize=8); ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=.3)
ax[0].set_title("Clean urban random tiles (built-up ≥40 %, snow/cloud <10 %)")
g = ["clean urban", "snow/cloud ≥10 %"]
ax[1].bar(np.arange(2) - .2, [summary_all["kate_on_clean"], summary_all["kate_on_snow_cloud"]], .4, label="KATE-CD model")
ax[1].bar(np.arange(2) + .2, [summary_all["xbd_on_clean"], summary_all["xbd_on_snow_cloud"]], .4, label="xBD model")
ax[1].set_xticks(range(2), [f"{g[0]} (n={summary_all['n_clean']})", f"{g[1]} (n={summary_all['n_snow_cloud']})"]); ax[1].legend(); ax[1].grid(axis="y", alpha=.3)
ax[1].set_title("Failure mode: snow and cloud produce false 'damage'")
plt.tight_layout(); plt.savefig(OUT / "city_summary.png", dpi=100); plt.close()

COL = np.array([[0, 0, 0], [60, 180, 75], [255, 225, 25], [245, 130, 48], [230, 25, 75]], np.uint8)
show = []
for c in cs:
    show += sorted([v for v in vis if v[0]["city"] == c and v[0]["clean_urban"]], key=lambda v: -v[3].mean())[:2]
show += sorted([v for v in vis if v[0]["snow_cloud_frac"] >= .1], key=lambda v: -v[3].mean())[:2]
fig, ax = plt.subplots(len(show), 4, figsize=(14, 3.6 * len(show)))
for r, (info, a, p, k, cls) in enumerate(show):
    tag = "" if info["clean_urban"] else f" [snow/cloud {100 * info['snow_cloud_frac']:.0f} %]"
    ax[r, 0].imshow(a); ax[r, 0].set_title(f"{info['city']}{tag} — pre {info['pre']}", fontsize=9)
    ax[r, 1].imshow(p); ax[r, 1].set_title(f"post {info['post']}", fontsize=9)
    ov = p.copy(); ov[k] = (0.5 * ov[k] + [127, 0, 0]).astype(np.uint8)
    ax[r, 2].imshow(ov); ax[r, 2].set_title("KATE-CD model (red = damage)", fontsize=9)
    ax[r, 3].imshow(COL[cls]); ax[r, 3].set_title("xBD model (green intact → red destroyed)", fontsize=9)
    for q in ax[r]: q.axis("off")
plt.tight_layout(); plt.savefig(OUT / "samples.png", dpi=65); plt.close()
