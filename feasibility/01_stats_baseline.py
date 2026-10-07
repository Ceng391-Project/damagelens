import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab
from skimage.exposure import match_histograms
from skimage.filters import gaussian

from common import OUT, load_kate, scores

OUT.mkdir(exist_ok=True)
splits = {s: load_kate(s) for s in ["train", "validation", "test"]}

stats = {}
for s, d in splits.items():
    pos = np.array([m.mean() for _, _, m in d])
    stats[s] = dict(n=len(d), damaged_pixel_ratio=float(pos.mean()), tiles_with_damage=float((pos > 0).mean()))
print(json.dumps(stats, indent=2))


def diff_map(pre, post):
    pre = match_histograms(pre, post, channel_axis=-1)
    d = np.linalg.norm(rgb2lab(post) - rgb2lab(pre), axis=-1)
    return gaussian(d, sigma=3)


def evaluate(data, thr):
    pred = np.concatenate([(diff_map(a, b) > thr).ravel() for a, b, _ in data])
    gt = np.concatenate([m.ravel() for _, _, m in data])
    return scores(pred, gt)


thresholds = np.arange(5, 61, 5)
val = {float(t): evaluate(splits["validation"], t)["f1"] for t in thresholds}
best = max(val, key=val.get)
test = evaluate(splits["test"], best)
print("val f1 by threshold", {k: round(v, 3) for k, v in val.items()})
print("best thr", best, "test", {k: round(float(v), 3) for k, v in test.items()})
json.dump(dict(stats=stats, baseline_val_f1=val, baseline_thr=best,
               baseline_test={k: float(v) for k, v in test.items()}),
          open(OUT / "01_stats_baseline.json", "w"), indent=2)

idx = np.argsort([-m.mean() for _, _, m in splits["test"]])[:4]
fig, ax = plt.subplots(4, 4, figsize=(12, 12))
for row, i in enumerate(idx):
    pre, post, m = splits["test"][i]
    for col, (im, t) in enumerate([(pre, "pre"), (post, "post"), (m, "label"), (diff_map(pre, post) > best, "diff baseline")]):
        ax[row, col].imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax[row, col].set_title(t); ax[row, col].axis("off")
plt.tight_layout(); plt.savefig(OUT / "01_kate_samples.png", dpi=80)
