import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from seg import binary_scores, predict, train_binary

ROOT = Path(__file__).parent
D = ROOT.parent.parent / "data" / "sen1floods11" / "sen1floods11_v1.1"
OUT = ROOT / "outputs" / "flood"
OUT.mkdir(parents=True, exist_ok=True)


def ids(split):
    return [l.split(",")[0].strip() for l in open(D / "splits" / f"flood_{split}_data.txt") if l.strip()]


def load(cid):
    s2 = rasterio.open(D / "data/S2L1CHand" / f"{cid}_S2Hand.tif").read().astype(np.float16) / 10000
    s1 = np.nan_to_num(rasterio.open(D / "data/S1GRDHand" / f"{cid}_S1Hand.tif").read(), nan=-30).clip(-40, 5).astype(np.float16)
    y = rasterio.open(D / "data/LabelHand" / f"{cid}_LabelHand.tif").read()[0]
    jrc = rasterio.open(D / "data/JRCWaterHand" / f"{cid}_JRCWaterHand.tif").read()[0]
    y = np.where(y < 0, 255, y).astype(np.uint8)
    return s2, s1, y, jrc


def split_arrays(split):
    c = ids(split)
    s2, s1, y, j = zip(*[load(i) for i in c])
    return c, np.stack(s2), np.stack(s1), np.stack(y), np.stack(j)


data = {s: split_arrays(s) for s in ["train", "valid", "test", "bolivia"]}
tr = data["train"]
m2, sd2 = tr[1].astype(np.float32).mean((0, 2, 3), keepdims=True), tr[1].astype(np.float32).std((0, 2, 3), keepdims=True) + 1e-6
m1, sd1 = tr[2].astype(np.float32).mean((0, 2, 3), keepdims=True), tr[2].astype(np.float32).std((0, 2, 3), keepdims=True) + 1e-6
feats = {
    "S2": lambda s: ((data[s][1] - m2) / sd2).astype(np.float16),
    "S1": lambda s: ((data[s][2] - m1) / sd1).astype(np.float16),
    "S1+S2": lambda s: np.concatenate([(data[s][2] - m1) / sd1, (data[s][1] - m2) / sd2], 1).astype(np.float16),
}


def mndwi(s):
    b3, b11 = data[s][1][:, 2].astype(np.float32), data[s][1][:, 11].astype(np.float32)
    return (b3 - b11) / (b3 + b11 + 1e-6)


def ndwi(s):
    b3, b8 = data[s][1][:, 2].astype(np.float32), data[s][1][:, 7].astype(np.float32)
    return (b3 - b8) / (b3 + b8 + 1e-6)


def tune(score_fn, split, grid, greater=True):
    s, y = score_fn(split), data[split][3]
    f = {float(t): binary_scores((s > t) if greater else (s < t), y == 1, y != 255)["f1"] for t in grid}
    return max(f, key=f.get)


preds = {}
results = {}
thr = {"NDWI": tune(ndwi, "valid", np.arange(-0.4, 0.41, 0.05)),
       "MNDWI": tune(mndwi, "valid", np.arange(-0.4, 0.41, 0.05)),
       "S1 VV threshold": tune(lambda s: data[s][2][:, 0], "valid", np.arange(-25, -9, 1), greater=False)}
for s in ["test", "bolivia"]:
    preds[("NDWI", s)] = ndwi(s) > thr["NDWI"]
    preds[("MNDWI", s)] = mndwi(s) > thr["MNDWI"]
    preds[("S1 VV threshold", s)] = data[s][2][:, 0] < thr["S1 VV threshold"]

histories = {}
for name, fx in feats.items():
    print("training", name, flush=True)
    model, hist = train_binary(fx("train"), data["train"][3], fx("valid"), data["valid"][3], epochs=30, crop=256)
    histories[f"U-Net {name}"] = hist
    for s in ["test", "bolivia"]:
        preds[(f"U-Net {name}", s)] = predict(model, fx(s)) > 0.5
    torch.save(model.state_dict(), OUT / f"unet_{name.replace('+', '_')}.pt")

methods = ["NDWI", "MNDWI", "S1 VV threshold", "U-Net S1", "U-Net S2", "U-Net S1+S2"]
for m in methods:
    for s in ["test", "bolivia"]:
        y, jrc = data[s][3], data[s][4]
        p = preds[(m, s)]
        all_ = binary_scores(p, y == 1, y != 255)
        flood_only = binary_scores(p, y == 1, (y != 255) & (jrc == 0))
        per_country = defaultdict(list)
        for k, cid in enumerate(data[s][0]):
            per_country[cid.split("_")[0]].append(k)
        pc = {c: binary_scores(p[ix], y[ix] == 1, y[ix] != 255)["iou"] for c, ix in per_country.items()}
        results[f"{m}|{s}"] = dict(all=all_, flood_only=flood_only, per_country=pc)
json.dump(dict(thresholds=thr, results=results, histories=histories), open(OUT / "results.json", "w"), indent=2)

# charts
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a, s, title in [(ax[0], "test", "Test (11 ülke, 90 kare)"), (ax[1], "bolivia", "Bolivia (eğitimde görülmemiş olay)")]:
    w = 0.38; x = np.arange(len(methods))
    a.bar(x - w / 2, [results[f"{m}|{s}"]["all"]["iou"] for m in methods], w, label="Tüm su (IoU)", color="#3b6ea5")
    a.bar(x + w / 2, [results[f"{m}|{s}"]["flood_only"]["iou"] for m in methods], w, label="Sadece taşkın suyu (IoU)", color="#e07b39")
    a.set_xticks(x, methods, rotation=25, ha="right"); a.set_ylim(0, 1); a.set_title(title); a.grid(axis="y", alpha=.3)
    for i, m in enumerate(methods):
        a.text(i - w / 2, results[f"{m}|{s}"]["all"]["iou"] + .01, f'{results[f"{m}|{s}"]["all"]["iou"]:.2f}', ha="center", fontsize=8)
ax[0].legend(loc="upper left")
plt.tight_layout(); plt.savefig(OUT / "iou_by_method.png", dpi=110); plt.close()

countries = sorted(results["U-Net S1+S2|test"]["per_country"])
M = np.array([[results[f"{m}|test"]["per_country"][c] for c in countries] for m in methods])
fig, a = plt.subplots(figsize=(10, 4))
im = a.imshow(M, cmap="viridis", vmin=0, vmax=1)
a.set_xticks(range(len(countries)), countries, rotation=30, ha="right"); a.set_yticks(range(len(methods)), methods)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < .6 else "k", fontsize=8)
plt.colorbar(im, label="IoU"); a.set_title("Ülke/olay bazında IoU (test)")
plt.tight_layout(); plt.savefig(OUT / "iou_by_country.png", dpi=110); plt.close()

fig, a = plt.subplots(figsize=(7, 4))
for k, h in histories.items():
    a.plot([e["val_f1"] for e in h], label=k)
a.set_xlabel("epoch"); a.set_ylabel("val F1"); a.legend(); a.grid(alpha=.3); a.set_title("Eğitim eğrileri")
plt.tight_layout(); plt.savefig(OUT / "training_curves.png", dpi=110); plt.close()

s = "test"
cand = [k for k, c in enumerate(data[s][0]) if 0.15 < (data[s][3][k] == 1).mean() < 0.6]
rng = np.random.default_rng(1); pick = rng.choice(cand, 4, replace=False)
fig, ax = plt.subplots(4, 5, figsize=(15, 12))
for r, k in enumerate(pick):
    rgb = np.clip(data[s][1][k][[3, 2, 1]].astype(np.float32).transpose(1, 2, 0) * 3.5, 0, 1)
    y = data[s][3][k].astype(float); y[y == 255] = np.nan
    panels = [(rgb, f"S2 RGB — {data[s][0][k]}"), (data[s][2][k][0].astype(np.float32), "S1 VV (dB)"), (y, "Etiket"),
              (preds[("MNDWI", s)][k], "MNDWI eşik"), (preds[("U-Net S1+S2", s)][k], "U-Net S1+S2")]
    for c, (im, t) in enumerate(panels):
        ax[r, c].imshow(im, cmap=None if im.ndim == 3 else ("gray" if c == 1 else "Blues"))
        ax[r, c].set_title(t, fontsize=9); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(OUT / "samples.png", dpi=75); plt.close()
print(json.dumps({k: round(v["all"]["iou"], 3) for k, v in results.items()}, indent=1))
