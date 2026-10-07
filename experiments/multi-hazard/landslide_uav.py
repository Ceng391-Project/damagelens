import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from seg import binary_scores, predict, train_binary

ROOT = Path(__file__).parent
D = ROOT.parent.parent / "data" / "landslide_uav"
OUT = ROOT / "outputs" / "landslide_uav"
OUT.mkdir(parents=True, exist_ok=True)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def load(prefix):
    X, Y = [], []
    for f in sorted(D.glob(f"{prefix}-*.parquet")):
        for r in pq.read_table(f).to_pylist():
            X.append(np.array(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").resize((384, 384), Image.BOX)))
            Y.append(np.array(Image.open(io.BytesIO(r["annotation"]["bytes"])).resize((384, 384), Image.NEAREST)))
    return np.stack(X), (np.stack(Y) > 0).astype(np.uint8)


norm = lambda X: ((X / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float16)
tr, va, te = load("train"), load("validation"), load("test")
print({k: (len(v[0]), float(v[1].mean())) for k, v in dict(train=tr, val=va, test=te).items()}, flush=True)

# colour/texture baseline: bare-soil index on RGB (low greenness, high brightness), tuned on val
def soil(X):
    X = X.astype(np.float32) / 255; r, g, b = X[..., 0], X[..., 1], X[..., 2]
    return (r - g) / (r + g + 1e-6) + 0.5 * (X.mean(-1) - 0.5)
best = max((binary_scores(soil(va[0]) > t, va[1])["f1"], float(t)) for t in np.arange(-0.3, 0.4, 0.025))
model, hist = train_binary(norm(tr[0]), tr[1], norm(va[0]), va[1], epochs=20, bs=8, pos_weight=2.0)
prob = predict(model, norm(te[0]))
preds = {"RGB soil index (rule)": soil(te[0]) > best[1], "U-Net RGB (UAV)": prob > 0.5}
res = {k: binary_scores(v, te[1]) for k, v in preds.items()}
json.dump(dict(results=res, baseline_thr=best[1], hist=hist, n=dict(train=len(tr[0]), val=len(va[0]), test=len(te[0])),
               landslide_px_frac=float(te[1].mean())), open(OUT / "results.json", "w"), indent=2)
print(json.dumps(res, indent=1))

per_img = np.array([binary_scores(preds["U-Net RGB (UAV)"][i], te[1][i])["iou"] for i in range(len(te[1])) if te[1][i].any()])
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
names = list(res); x = np.arange(len(names)); w = .2
for j, m in enumerate(["precision", "recall", "f1", "iou"]):
    ax[0].bar(x + (j - 1.5) * w, [res[n][m] for n in names], w, label=m)
ax[0].set_xticks(x, names); ax[0].set_ylim(0, 1); ax[0].legend(); ax[0].grid(axis="y", alpha=.3); ax[0].set_title(f"UAV landslide test ({len(te[0])} chips)")
ax[1].hist(per_img, bins=20, color="#3b6ea5"); ax[1].set_xlabel("IoU per chip"); ax[1].set_title("U-Net: spread across chips"); ax[1].grid(alpha=.3)
ax[2].plot([h["val_f1"] for h in hist]); ax[2].set_xlabel("epoch"); ax[2].set_ylabel("val F1"); ax[2].grid(alpha=.3); ax[2].set_title("Training curve")
plt.tight_layout(); plt.savefig(OUT / "metrics.png", dpi=100); plt.close()
rng = np.random.default_rng(5); pick = rng.choice(np.where(te[1].mean((1, 2)) > .05)[0], 4, replace=False)
fig, ax = plt.subplots(4, 4, figsize=(13, 13))
for r, i in enumerate(pick):
    for c, (im, t) in enumerate([(te[0][i], "UAV image"), (te[1][i], "Label"), (preds["RGB soil index (rule)"][i], "Rule"), (preds["U-Net RGB (UAV)"][i], "U-Net")]):
        ax[r, c].imshow(im, cmap=None if im.ndim == 3 else "Reds"); ax[r, c].set_title(t, fontsize=9); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(OUT / "samples.png", dpi=70); plt.close()
