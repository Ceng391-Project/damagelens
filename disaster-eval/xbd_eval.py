import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from scipy import ndimage

from seg import DEV, binary_scores
from xbd_train import load, tier3_split

ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "xbd"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "xbd_unet5.pt"
TYPE = {
    "mexico-earthquake": "earthquake", "palu-tsunami": "tsunami", "sunda-tsunami": "tsunami",
    "midwest-flooding": "flood", "nepal-flooding": "flood",
    "hurricane-harvey": "hurricane", "hurricane-florence": "hurricane", "hurricane-michael": "hurricane", "hurricane-matthew": "hurricane",
    "joplin-tornado": "tornado", "moore-tornado": "tornado", "tuscaloosa-tornado": "tornado",
    "socal-fire": "fire", "santa-rosa-wildfire": "fire", "woolsey-fire": "fire", "portugal-wildfire": "fire", "pinery-bushfire": "fire",
    "guatemala-volcano": "volcano", "lower-puna-volcano": "volcano",
}
CLS = ["arka plan", "hasarsız", "az hasar", "ağır hasar", "yıkılmış"]
COL = np.array([[0, 0, 0], [60, 180, 75], [255, 225, 25], [245, 130, 48], [230, 25, 75]], np.uint8)

model = smp.Unet("resnet18", encoder_weights=None, in_channels=6, classes=5).to(DEV)
model.load_state_dict(torch.load(OUT / CKPT, map_location=DEV)); model.eval()
mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=DEV).view(1, 6, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225] * 2, device=DEV).view(1, 6, 1, 1)


@torch.no_grad()
def infer(x):  # x uint8 (B,H,W,6)
    t = (torch.from_numpy(np.ascontiguousarray(x)).to(DEV).permute(0, 3, 1, 2).float() / 255 - mean) / std
    lo = model(t).softmax(1) + model(t.flip(-1)).softmax(1).flip(-1)
    return lo.cpu().numpy()


te = load("test"); t3 = load("tier3"); part = tier3_split()
items = [(te, i) for i in range(len(te[2]))] + [(t3, i) for i in np.where(part == 2)[0]]
pix = defaultdict(lambda: np.zeros((5, 5), np.int64))
bld = defaultdict(lambda: np.zeros((5, 5), np.int64))
examples = defaultdict(list)
rng = np.random.default_rng(7)
for k in range(0, len(items), 8):
    chunk = items[k:k + 8]
    x = np.stack([s[0][i] for s, i in chunk]); y = np.stack([s[1][i] for s, i in chunk])
    pr = infer(x); p = pr.argmax(1)
    for j, (s, i) in enumerate(chunk):
        ev = s[2][i]; yy, pp = y[j], p[j]; m = yy < 5
        pix[ev] += np.bincount(yy[m] * 5 + pp[m], minlength=25).reshape(5, 5)
        lab, n = ndimage.label((yy >= 1) & (yy <= 4))
        for c in range(1, n + 1):
            reg = lab == c
            g = np.bincount(yy[reg], minlength=5)[1:].argmax() + 1
            pd_ = pp[reg]; pd_ = pd_[pd_ > 0]
            q = np.bincount(pd_, minlength=5)[1:].argmax() + 1 if len(pd_) > .2 * reg.sum() else 0
            bld[ev][g, q] += 1
        if len(examples[TYPE[ev]]) < 3 and (yy >= 2).mean() > 0.01 and rng.random() < .5:
            examples[TYPE[ev]].append((ev, x[j], yy, pp))
    if k % 200 == 0:
        print(k, len(items), flush=True)


def metrics(cm, b):
    tp = cm[1:, 1:].sum(); loc = 2 * tp / max(2 * tp + cm[0, 1:].sum() + cm[1:, 0].sum(), 1)
    f1 = []
    for c in range(1, 5):
        t = cm[c, c]; denom = 2 * t + (cm[1:, c].sum() - t) + (cm[c, 1:].sum() - t)
        f1.append(2 * t / denom if cm[c, 1:].sum() > 0 else np.nan)
    v = [f for f in f1 if not np.isnan(f)]
    dmg = len(v) / sum(1 / max(f, 1e-6) for f in v) if v else np.nan
    gt_dmg = b[2:, :].sum(); pr_dmg = b[1:, 2:].sum(); tpd = b[2:, 2:].sum()
    bin_f1 = 2 * tpd / max(gt_dmg + pr_dmg, 1)
    return dict(loc_f1=float(loc), dmg_f1_per_class=[None if np.isnan(f) else float(f) for f in f1], dmg_f1=float(dmg),
                xview2=float(.3 * loc + .7 * dmg), buildings=int(b[1:].sum()), building_detect_rate=float(b[1:, 1:].sum() / max(b[1:].sum(), 1)),
                building_acc=float(np.trace(b[1:, 1:]) / max(b[1:].sum(), 1)), building_damaged_f1=float(bin_f1),
                destroyed_recall=float(b[4, 4] / max(b[4].sum(), 1)), gt_damaged_share=float(gt_dmg / max(b[1:].sum(), 1)))


by_type_pix = defaultdict(lambda: np.zeros((5, 5), np.int64)); by_type_bld = defaultdict(lambda: np.zeros((5, 5), np.int64))
for ev in pix:
    by_type_pix[TYPE[ev]] += pix[ev]; by_type_bld[TYPE[ev]] += bld[ev]
res = dict(per_event={ev: dict(type=TYPE[ev], **metrics(pix[ev], bld[ev])) for ev in sorted(pix)},
           per_type={t: metrics(by_type_pix[t], by_type_bld[t]) for t in sorted(by_type_pix)},
           building_confusion={t: by_type_bld[t].tolist() for t in by_type_bld}, n_test_images=len(items))
tot_p = sum(by_type_pix.values()); tot_b = sum(by_type_bld.values()); res["overall"] = metrics(tot_p, tot_b)

# 6 Şubat zero-shot on KATE-CD (binary: model class >= 2 vs damage polygons), two input scales
kate = pq.read_table(ROOT.parent / "feasibility" / "data" / "kate-cd" / "test.parquet").to_pylist()
dec = lambda c: np.array(Image.open(io.BytesIO(c["bytes"])).convert("RGB"))
kx = np.stack([np.concatenate([dec(r["pre_image"]), dec(r["post_image"])], -1) for r in kate])
ky = np.stack([(np.array(Image.open(io.BytesIO(r["label"]["bytes"]))) > 0) for r in kate])
res["kate_zero_shot"] = {}
kate_pred = {}
for scale in [512, 256]:
    xs = kx if scale == 512 else np.stack([np.array(Image.fromarray(a[..., :3]).resize((256, 256), Image.BOX)) for a in kx])
    if scale == 256:
        xs = np.concatenate([xs, np.stack([np.array(Image.fromarray(a[..., 3:]).resize((256, 256), Image.BOX)) for a in kx])], -1)
    probs = np.concatenate([infer(xs[i:i + 8]) for i in range(0, len(xs), 8)])
    dmg = probs[:, 2:].sum(1) / 2
    if scale == 256:
        dmg = torch.nn.functional.interpolate(torch.from_numpy(dmg)[:, None], size=512, mode="bilinear")[:, 0].numpy()
    best = max(((binary_scores(dmg > t, ky)["f1"], t) for t in np.arange(.1, .9, .1)))
    res["kate_zero_shot"][f"scale_{scale}"] = dict(f1_at_0_5=binary_scores(dmg > .5, ky)["f1"], best_f1_oracle_thr=best[0], oracle_thr=float(best[1]),
                                                   **{k: v for k, v in binary_scores(dmg > .5, ky).items()})
    kate_pred[scale] = dmg
json.dump(res, open(OUT / f"eval_{CKPT.replace('.pt', '')}.json", "w"), indent=2)
print(json.dumps({t: {k: v for k, v in m.items() if k != "dmg_f1_per_class"} for t, m in res["per_type"].items()}, indent=1))
print(json.dumps(res["kate_zero_shot"], indent=1))

if CKPT != "xbd_unet5.pt":
    sys.exit()
types = [t for t in ["earthquake", "tsunami", "flood", "hurricane", "tornado", "fire", "volcano"] if t in res["per_type"]]
fig, ax = plt.subplots(1, 2, figsize=(17, 5))
x = np.arange(len(types)); w = .2
for j, (k, lab) in enumerate([("loc_f1", "Bina bulma F1"), ("dmg_f1", "Hasar sınıf F1 (harmonik)"), ("xview2", "xView2 skoru"), ("building_damaged_f1", "Bina bazlı hasarlı/hasarsız F1")]):
    ax[0].bar(x + (j - 1.5) * w, [res["per_type"][t][k] for t in types], w, label=lab)
ax[0].set_xticks(x, types); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=.3); ax[0].set_title("xBD test: afet türüne göre performans")
evs = sorted(res["per_event"], key=lambda e: (TYPE[e], e))
cm_ = plt.get_cmap("tab10"); tc = {t: cm_(i) for i, t in enumerate(types)}
ax[1].bar(range(len(evs)), [res["per_event"][e]["building_damaged_f1"] for e in evs], color=[tc[TYPE[e]] for e in evs])
ax[1].set_xticks(range(len(evs)), evs, rotation=60, ha="right", fontsize=8); ax[1].set_ylim(0, 1); ax[1].grid(axis="y", alpha=.3)
ax[1].set_title("Olay bazında bina hasarlı/hasarsız F1 (renk = afet türü)")
plt.tight_layout(); plt.savefig(OUT / "per_type.png", dpi=100); plt.close()

fig, ax = plt.subplots(1, len(types), figsize=(4 * len(types), 4))
for a, t in zip(ax, types):
    b = by_type_bld[t][1:, :].astype(float); b = b / np.maximum(b.sum(1, keepdims=True), 1)
    a.imshow(b, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(5):
            a.text(j, i, f"{b[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if b[i, j] > .5 else "k")
    a.set_xticks(range(5), ["kaçan"] + CLS[1:], rotation=45, ha="right", fontsize=7); a.set_yticks(range(4), CLS[1:], fontsize=7)
    a.set_title(f"{t} (n={int(by_type_bld[t][1:].sum())})", fontsize=9); a.set_xlabel("tahmin", fontsize=7)
ax[0].set_ylabel("gerçek")
plt.tight_layout(); plt.savefig(OUT / "confusion_by_type.png", dpi=100); plt.close()

for t in types:
    ex = examples[t]
    if not ex:
        continue
    fig, ax = plt.subplots(len(ex), 4, figsize=(13, 3.3 * len(ex)), squeeze=False)
    for r, (ev, xx, yy, pp) in enumerate(ex):
        ax[r, 0].imshow(xx[..., :3]); ax[r, 0].set_title(f"{ev} — öncesi", fontsize=9)
        ax[r, 1].imshow(xx[..., 3:]); ax[r, 1].set_title("sonrası", fontsize=9)
        ax[r, 2].imshow(COL[np.where(yy < 5, yy, 0)]); ax[r, 2].set_title("gerçek hasar", fontsize=9)
        ax[r, 3].imshow(COL[pp]); ax[r, 3].set_title("model tahmini", fontsize=9)
        for a in ax[r]: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / f"samples_{t}.png", dpi=70); plt.close()

idx = np.argsort(-ky.mean((1, 2)))[:3]
fig, ax = plt.subplots(3, 5, figsize=(16, 10))
for r, i in enumerate(idx):
    pan = [(kx[i][..., :3], "öncesi"), (kx[i][..., 3:], "sonrası"), (ky[i], "KATE-CD etiketi"),
           (kate_pred[512][i] > .5, "xBD modeli, 512 ölçek"), (kate_pred[256][i] > .5, "xBD modeli, 256 ölçek")]
    for c, (im, tt) in enumerate(pan):
        ax[r, c].imshow(im, cmap=None if im.ndim == 3 else "gray"); ax[r, c].set_title(tt, fontsize=9); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(OUT / "kate_zero_shot.png", dpi=70); plt.close()
