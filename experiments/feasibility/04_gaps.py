import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from scipy.ndimage import shift as nd_shift, uniform_filter
from skimage.color import rgb2gray, rgb2lab
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu, gaussian
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from common import OUT, load_kate, scores

dev = "mps" if torch.backends.mps.is_available() else "cpu"
test, val = load_kate("test"), load_kate("validation")
res = {}

# ---------- A. classical change-detection baselines ----------
def cva(pre, post):
    pre = match_histograms(pre, post, channel_axis=-1)
    return gaussian(np.linalg.norm(rgb2lab(post) - rgb2lab(pre), axis=-1), 3)


def ssim_change(pre, post):
    _, s = structural_similarity(rgb2gray(pre), rgb2gray(post), full=True, data_range=1.0, win_size=11)
    return gaussian(1 - s, 3)


def pca_kmeans(pre, post, h=4, s=3):  # Celik 2009
    d = np.abs(rgb2gray(post) - rgb2gray(match_histograms(pre, post, channel_axis=-1)))
    pad = np.pad(d, h, mode="reflect")
    blocks = np.lib.stride_tricks.sliding_window_view(pad, (2 * h + 1, 2 * h + 1)).reshape(d.size, -1)
    pca = PCA(s).fit(blocks[::17]); f = pca.transform(blocks)
    km = KMeans(2, n_init=3, random_state=0).fit(f[::17]); lab = km.predict(f).reshape(d.shape)
    hi = np.argmax([d[lab == k].mean() for k in range(2)])
    return gaussian((lab == hi).astype(float), 1)


def evaluate_map(fn, data, thr):
    P = np.concatenate([(fn(a, b) > thr).ravel() for a, b, _ in data]); G = np.concatenate([m.ravel() for _, _, m in data])
    return {k: float(v) for k, v in scores(P, G).items()}


def tune(fn, grid):
    vals = {float(t): evaluate_map(fn, val, t)["f1"] for t in grid}
    return max(vals, key=vals.get)


import os
prev = json.load(open(OUT / "04_gaps.json")) if os.path.exists(OUT / "04_gaps.json") else None
baselines = {}
if prev: baselines = prev["baselines"]
else:
  t = tune(cva, np.arange(5, 41, 5)); baselines["CVA (Lab büyüklüğü)"] = evaluate_map(cva, test, t) | {"thr": t}
  t = tune(ssim_change, np.arange(.1, .9, .1)); baselines["1−SSIM yapısal fark"] = evaluate_map(ssim_change, test, t) | {"thr": t}
  baselines["PCA-kmeans (Celik 2009)"] = evaluate_map(pca_kmeans, test, 0.5) | {"thr": 0.5}
  otsu = lambda a, b: (lambda c: c > threshold_otsu(c))(cva(a, b)).astype(float)
  baselines["CVA + Otsu (eşik ayarsız)"] = evaluate_map(otsu, test, 0.5) | {"thr": "otsu"}
res["baselines"] = baselines
print(json.dumps(baselines, indent=1), flush=True)

# ---------- B. alignment sensitivity + registration recovery ----------
model = smp.Unet("resnet18", encoder_weights=None, in_channels=6, classes=1).to(dev)
model.load_state_dict(torch.load(OUT / "kate_only.pt", map_location=dev)); model.eval()
thr = json.load(open(OUT / "kate_only.json"))["thr"]
mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=dev).view(1, 6, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225] * 2, device=dev).view(1, 6, 1, 1)


@torch.no_grad()
def unet(pairs):
    out = []
    for i in range(0, len(pairs), 8):
        x = np.stack([np.concatenate([a, b], -1) for a, b in pairs[i:i + 8]]).transpose(0, 3, 1, 2)
        out.append(torch.sigmoid(model((torch.from_numpy(x).to(dev).float() / 255 - mean) / std)).cpu().numpy()[:, 0])
    return np.concatenate(out)


def shifted(img, d):
    return np.clip(nd_shift(img.astype(np.float32), (d, d * 0.6, 0), order=1, mode="nearest"), 0, 255).astype(np.uint8)


G = np.stack([m for _, _, m in test])
real_shift = [float(np.hypot(*phase_cross_correlation(rgb2gray(a), rgb2gray(b), upsample_factor=4)[0])) for a, b, _ in test]
res["kate_pairs_estimated_misregistration_px"] = dict(median=float(np.median(real_shift)), p90=float(np.percentile(real_shift, 90)), max=float(np.max(real_shift)))
align = {}
for d in [0, 2, 4, 8, 16, 32]:
    pre_s = [shifted(a, d) for a, _, _ in test]  # labels live in the post frame, so misregister the pre image
    p = unet([(a, b) for a, (_, b, _) in zip(pre_s, test)]) > thr
    reg = []
    for a, (_, b, _) in zip(pre_s, test):
        sh = phase_cross_correlation(rgb2gray(b), rgb2gray(a), upsample_factor=4)[0]
        reg.append(np.clip(nd_shift(a.astype(np.float32), (sh[0], sh[1], 0), order=1, mode="nearest"), 0, 255).astype(np.uint8))
    pr = unet([(a, b) for a, (_, b, _) in zip(reg, test)]) > thr
    cv = np.stack([cva(a, b) > baselines["CVA (Lab büyüklüğü)"]["thr"] for a, (_, b, _) in zip(pre_s, test)])
    align[d] = dict(unet_f1=float(scores(p, G)["f1"]), unet_f1_after_registration=float(scores(pr, G)["f1"]), cva_f1=float(scores(cv, G)["f1"]))
    print(d, align[d], flush=True)
res["alignment"] = align
json.dump(res, open(OUT / "04_gaps.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
names = ["Görüntü farkı (01)"] + list(baselines) + ["U-Net (KATE-CD)"]
f01 = json.load(open(OUT / "01_stats_baseline.json"))["baseline_test"]["f1"]
f1s = [f01] + [baselines[k]["f1"] for k in baselines] + [json.load(open(OUT / "kate_only.json"))["test"]["f1"]]
ax[0].barh(names, f1s, color=["#9aa"] * (len(names) - 1) + ["#3b6ea5"]); ax[0].set_xlim(0, .7); ax[0].set_xlabel("F1 (KATE-CD test)")
for i, v in enumerate(f1s): ax[0].text(v + .01, i, f"{v:.2f}", va="center")
ax[0].set_title("Klasik değişim tespiti vs öğrenilmiş model"); ax[0].grid(axis="x", alpha=.3)
ds = list(align)
ax[1].plot(ds, [align[d]["unet_f1"] for d in ds], "o-", label="U-Net, kaydırılmış")
ax[1].plot(ds, [align[d]["unet_f1_after_registration"] for d in ds], "s--", label="U-Net, faz korelasyonu ile hizalandıktan sonra")
ax[1].plot(ds, [align[d]["cva_f1"] for d in ds], "^:", label="CVA, kaydırılmış")
ax[1].set_xscale("symlog", linthresh=2); ax[1].set_xticks(ds, [str(d) for d in ds]); ax[1].set_xlabel("yapay hizalama hatası (piksel, ~0,4 m/px)"); ax[1].set_ylabel("F1")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_title("Hizalama hatasına duyarlılık")
plt.tight_layout(); plt.savefig(OUT / "04_gaps.png", dpi=100); plt.close()
