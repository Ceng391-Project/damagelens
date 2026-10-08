import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from seg import binary_scores, predict, train_binary

ROOT = Path(__file__).parent
D = ROOT.parent.parent / "data" / "landslide4sense"
OUT = ROOT / "outputs" / "landslide"
OUT.mkdir(parents=True, exist_ok=True)


def load(split):
    n = len(list((D / "images" / split).glob("*.h5")))
    X = np.stack([h5py.File(D / "images" / split / f"image_{i}.h5")["img"][()].astype(np.float16) for i in range(1, n + 1)]).transpose(0, 3, 1, 2)
    Y = np.stack([h5py.File(D / "annotations" / split / f"mask_{i}.h5")["mask"][()] for i in range(1, n + 1)]).astype(np.uint8)
    return X, Y


data = {s: load(s) for s in ["train", "validation", "test"]}
Xtr = data["train"][0]
mu = Xtr.astype(np.float32).mean((0, 2, 3), keepdims=True); sd = Xtr.astype(np.float32).std((0, 2, 3), keepdims=True) + 1e-6
norm = lambda X, idx=None: ((X.astype(np.float32) - mu) / sd)[:, idx if idx is not None else slice(None)].astype(np.float16)
print({s: (len(v[1]), float(v[1].mean())) for s, v in data.items()}, flush=True)


def ndvi(X):
    r, n = X[:, 3].astype(np.float32), X[:, 7].astype(np.float32)
    return (n - r) / (n + r + 1e-6)


# rule baseline: bare (low NDVI) on steep slope
best = (-1, None)
Xv, Yv = data["validation"]
for t in np.arange(-0.2, 0.6, 0.05):
    for sl in np.quantile(Xv[:, 12].astype(np.float32), [0, .3, .5, .7]):
        f = binary_scores((ndvi(Xv) < t) & (Xv[:, 12] > sl), Yv == 1)["f1"]
        if f > best[0]:
            best = (f, (float(t), float(sl)))
t, sl = best[1]
preds, probs, hist = {}, {}, {}
Xt, Yt = data["test"]
preds["NDVI + slope rule"] = (ndvi(Xt) < t) & (Xt[:, 12] > sl)

for name, idx in [("U-Net RGB (Google-Earth-like)", [3, 2, 1]), ("U-Net 12-band S2", list(range(12))), ("U-Net S2 + slope + DEM", None)]:
    print("training", name, flush=True)
    model, h = train_binary(norm(Xtr, idx), data["train"][1], norm(Xv, idx), Yv, epochs=40, bs=32, pos_weight=3.0)
    hist[name] = h
    probs[name] = predict(model, norm(Xt, idx), bs=64)
    preds[name] = probs[name] > 0.5

res = {k: binary_scores(v, Yt == 1) for k, v in preds.items()}
json.dump(dict(rule=dict(ndvi_lt=t, slope_gt=sl), results=res, hist=hist,
               split_sizes={s: len(v[1]) for s, v in data.items()}, landslide_px_frac={s: float(v[1].mean()) for s, v in data.items()}),
          open(OUT / "results.json", "w"), indent=2)
print(json.dumps({k: {m: round(x, 3) for m, x in v.items()} for k, v in res.items()}, indent=1))

names = list(res)
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
x = np.arange(len(names)); w = .2
for j, m in enumerate(["precision", "recall", "f1", "iou"]):
    ax[0].bar(x + (j - 1.5) * w, [res[n][m] for n in names], w, label=m)
ax[0].set_xticks(x, names, rotation=15, ha="right"); ax[0].set_ylim(0, 1); ax[0].legend(); ax[0].grid(axis="y", alpha=.3)
ax[0].set_title("Landslide detection, Landslide4Sense test (800 chips)")
for n, p in probs.items():
    pr, rc, _ = precision_recall_curve((Yt == 1).ravel()[::7], p.ravel()[::7])
    ax[1].plot(rc, pr, label=n)
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_title("Precision–recall curves")
plt.tight_layout(); plt.savefig(OUT / "metrics.png", dpi=100); plt.close()

rng = np.random.default_rng(3)
cand = np.where(Yt.mean((1, 2)) > 0.05)[0]; pick = rng.choice(cand, 5, replace=False)
fig, ax = plt.subplots(5, 5, figsize=(15, 15))
for r, k in enumerate(pick):
    rgb = Xt[k][[3, 2, 1]].astype(np.float32).transpose(1, 2, 0); rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
    pan = [(rgb, "RGB"), (Yt[k], "Label"), (preds["NDVI + slope rule"][k], "NDVI+slope"),
           (preds["U-Net RGB (Google-Earth-like)"][k], "U-Net RGB"), (preds["U-Net S2 + slope + DEM"][k], "U-Net S2+slope+DEM")]
    for c, (im, tt) in enumerate(pan):
        ax[r, c].imshow(im, cmap=None if im.ndim == 3 else "Reds"); ax[r, c].set_title(tt, fontsize=9); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(OUT / "samples.png", dpi=75); plt.close()
