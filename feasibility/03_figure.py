import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch

from common import OUT, load_kate

test = load_kate("test")
idx = np.argsort([-m.mean() for _, _, m in test])[:4]
mean = torch.tensor([0.485, 0.456, 0.406] * 2).view(1, 6, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225] * 2).view(1, 6, 1, 1)
names = ["xbd_only", "kate_only", "xbd_then_kate"]
preds = {}
for n in names:
    m = smp.Unet("resnet18", encoder_weights=None, in_channels=6, classes=1)
    m.load_state_dict(torch.load(OUT / f"{n}.pt", map_location="cpu")); m.eval()
    thr = json.load(open(OUT / f"{n}.json"))["thr"]
    x = torch.from_numpy(np.stack([np.concatenate([test[i][0], test[i][1]], -1) for i in idx]).transpose(0, 3, 1, 2))
    with torch.no_grad():
        preds[n] = (torch.sigmoid(m((x.float() / 255 - mean) / std)) > thr).numpy()[:, 0]
fig, ax = plt.subplots(4, 6, figsize=(18, 12))
for r, i in enumerate(idx):
    pre, post, gt = test[i]
    for c, (im, t) in enumerate([(pre, "pre"), (post, "post"), (gt, "label")] + [(preds[n][r], n) for n in names]):
        ax[r, c].imshow(im, cmap="gray" if im.ndim == 2 else None); ax[r, c].set_title(t); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(OUT / "03_predictions.png", dpi=70)
