import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from rasterio.features import rasterize

from eo import CAT, grid, read_to_grid, s2_mosaic, worldcover
from seg import DEV, binary_scores

ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "flood_valencia"
OUT.mkdir(parents=True, exist_ok=True)
BBOX, RES = [-0.65, 39.15, -0.30, 39.48], 0.0001
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]  # Sen1Floods11 has B10 between B09 and B11
tr, W, H = grid(BBOX, RES)

gt_gdf = gpd.read_file(ROOT.parent.parent / "data/valencia/EMSR773_AOI01_DEL_PRODUCT_observedEventA_v1.json")
gt = rasterize([g for g in gt_gdf.geometry], out_shape=(H, W), transform=tr).astype(bool)
wc = worldcover(BBOX, RES)
perm = wc == 80

cache = OUT / "cache.npz"
if cache.exists():
    z = np.load(cache); s2 = z["s2"]; s1 = z["s1"]; s1_pre = z["s1_pre"]
else:
    b, ids = s2_mosaic(BBOX, "2024-10-31/2024-10-31", RES, bands=S2_BANDS, max_cloud=100)
    s2 = np.stack([b[k] for k in S2_BANDS]).astype(np.float32)

    def s1(date):
        its = list(CAT.search(collections=["sentinel-1-rtc"], bbox=BBOX, datetime=date).items())
        out = np.full((2, H, W), np.nan, np.float32)
        for it in its:
            for c, pol in enumerate(["vv", "vh"]):
                a = read_to_grid(it.assets[pol].href, BBOX, RES, nodata=0)
                m = np.isnan(out[c]) & np.isfinite(a) & (a > 0); out[c][m] = 10 * np.log10(a[m])
        return out
    s1, s1_pre = s1("2024-11-01/2024-11-01"), s1("2024-10-25/2024-10-25")
    np.savez_compressed(cache, s2=s2, s1=s1, s1_pre=s1_pre)

flood_thr = json.load(open(ROOT / "outputs/flood/results.json"))["thresholds"]
b3, b8, b11 = s2[2], s2[7], s2[10]
ndwi = (b3 - b8) / (b3 + b8 + 1e-6); mndwi = (b3 - b11) / (b3 + b11 + 1e-6)
s2_ok = np.isfinite(s2).all(0); s1_ok = np.isfinite(s1).all(0)

# Sen1Floods11-trained U-Nets applied without retraining
spec_stats = np.load(OUT / "s1f11_stats.npz") if (OUT / "s1f11_stats.npz").exists() else None
if spec_stats is None:
    import rasterio
    D = ROOT.parent.parent / "data/sen1floods11/sen1floods11_v1.1"
    ids = [l.split(",")[0].strip() for l in open(D / "splits/flood_train_data.txt") if l.strip()]
    acc = {k: [] for k in ("s2", "s1")}
    for i in ids:
        acc["s2"].append(rasterio.open(D / "data/S2L1CHand" / f"{i}_S2Hand.tif").read().astype(np.float32).reshape(13, -1)[:, ::16] / 10000)
        acc["s1"].append(np.nan_to_num(rasterio.open(D / "data/S1GRDHand" / f"{i}_S1Hand.tif").read(), nan=-30).clip(-40, 5).reshape(2, -1)[:, ::16])
    a2, a1 = np.concatenate(acc["s2"], 1), np.concatenate(acc["s1"], 1)
    np.savez(OUT / "s1f11_stats.npz", m2=a2.mean(1), s2=a2.std(1), m1=a1.mean(1), s1=a1.std(1))
    spec_stats = np.load(OUT / "s1f11_stats.npz")


def run_unet(path, x):
    m = smp.Unet("resnet18", encoder_weights=None, in_channels=x.shape[0], classes=1).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV)); m.eval()
    out = np.zeros(x.shape[1:], np.float32); T = 1024
    with torch.no_grad():
        for i in range(0, x.shape[1], T):
            for j in range(0, x.shape[2], T):
                t = x[:, i:i + T, j:j + T]; h, w = t.shape[1:]
                ph, pw = (-h) % 32, (-w) % 32
                tt = torch.from_numpy(np.pad(t, ((0, 0), (0, ph), (0, pw)))).to(DEV)[None]
                out[i:i + h, j:j + w] = torch.sigmoid(m(tt)).cpu().numpy()[0, 0, :h, :w]
    return out


st = spec_stats
s2_13 = np.insert(np.nan_to_num(s2, nan=0), 10, st["m2"][10], axis=0)  # B10 (cirrus, L1C only) set to training mean
x_s2 = ((s2_13 - st["m2"][:, None, None]) / st["s2"][:, None, None]).astype(np.float32)
s1c = np.nan_to_num(s1, nan=-30).clip(-40, 5)
x_s1 = ((s1c - st["m1"][:, None, None]) / st["s1"][:, None, None]).astype(np.float32)
p_unet_s2 = run_unet(ROOT / "outputs/flood/unet_S2.pt", x_s2) > .5
p_unet_s1 = run_unet(ROOT / "outputs/flood/unet_S1.pt", x_s1) > .5

preds = {
    "NDWI eşik (S2, 31 Eki)": (ndwi > flood_thr["NDWI"], s2_ok),
    "MNDWI eşik (S2, 31 Eki)": (mndwi > flood_thr["MNDWI"], s2_ok),
    "U-Net S2 (Sen1Floods11'den aktarım)": (p_unet_s2, s2_ok),
    "S1 VV eşik (1 Kas)": (s1[0] < flood_thr["S1 VV threshold"], s1_ok),
    "S1 değişim VV(1 Kas)−VV(25 Eki) < −3 dB": ((s1[0] - s1_pre[0]) < -3, s1_ok & np.isfinite(s1_pre[0])),
    "U-Net S1 (Sen1Floods11'den aktarım)": (p_unet_s1, s1_ok),
}
res = {}
for k, (p, ok) in preds.items():
    ev = ok & ~perm
    sc = binary_scores(p, gt, ev)
    urb = ev & (wc == 50)
    sc["recall_urban"] = binary_scores(p, gt, urb)["recall"]
    sc["recall_cropland"] = binary_scores(p, gt, ev & (wc == 40))["recall"]
    sc["valid_fraction"] = float(ev.mean())
    res[k] = sc
res_meta = dict(gt_flood_km2=float(gt.sum() * (RES * 111.32) ** 2 * np.cos(np.radians(39.3))), bbox=BBOX,
                gt_source="Copernicus EMS EMSR773 AOI01 DEL_PRODUCT v1 observedEvent (from Landsat-8 30 Oct + Sentinel-2 31 Oct)",
                thresholds_from="Sen1Floods11 validation split", results=res)
json.dump(res_meta, open(OUT / "results.json", "w"), indent=2)
print(json.dumps({k: {m: round(v, 3) for m, v in r.items()} for k, r in res.items()}, indent=1))

names = list(res)
fig, ax = plt.subplots(1, 2, figsize=(17, 5))
x = np.arange(len(names)); w = .2
for j, m in enumerate(["precision", "recall", "f1", "iou"]):
    ax[0].bar(x + (j - 1.5) * w, [res[n][m] for n in names], w, label=m)
ax[0].set_xticks(x, names, rotation=20, ha="right", fontsize=8); ax[0].set_ylim(0, 1); ax[0].legend(); ax[0].grid(axis="y", alpha=.3)
ax[0].set_title("Valencia DANA (29 Eki 2024) — Copernicus EMS taşkın sınırına göre")
ax[1].bar(x - .2, [res[n]["recall_cropland"] for n in names], .4, label="tarım alanı recall")
ax[1].bar(x + .2, [res[n]["recall_urban"] for n in names], .4, label="yerleşim alanı recall")
ax[1].set_xticks(x, names, rotation=20, ha="right", fontsize=8); ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].grid(axis="y", alpha=.3)
ax[1].set_title("Kentsel taşkın uydudan zor görünür")
plt.tight_layout(); plt.savefig(OUT / "metrics.png", dpi=100); plt.close()

ext = [BBOX[0], BBOX[2], BBOX[1], BBOX[3]]
rgb = np.clip(np.nan_to_num(s2[[3, 2, 1]]).transpose(1, 2, 0) * 3.5, 0, 1)
fig, ax = plt.subplots(1, 4, figsize=(22, 6))
ax[0].imshow(rgb, extent=ext); ax[0].set_title("Sentinel-2, 31 Eki 2024")
ax[1].imshow(np.nan_to_num(s1[0], nan=-30), cmap="gray", vmin=-25, vmax=0, extent=ext); ax[1].set_title("Sentinel-1 VV (dB), 1 Kas 2024")
ax[2].imshow(gt & ~perm, cmap="Blues", extent=ext); ax[2].set_title("Copernicus EMS taşkın alanı (yer gerçeği)")
best = max(res, key=lambda k: res[k]["f1"]); p, ok = preds[best]
ov = np.ones((H, W, 3)) * .95; ev = ok & ~perm
ov[p & gt & ev] = [0, .6, 0]; ov[p & ~gt & ev] = [.9, .2, .2]; ov[~p & gt & ev] = [.2, .3, .9]; ov[~ok] = [.7, .7, .7]
ax[3].imshow(ov, extent=ext); ax[3].set_title(f"En iyi: {best}\nyeşil=doğru, kırmızı=yanlış alarm, mavi=kaçan, gri=bulut/veri yok", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "maps.png", dpi=70); plt.close()
