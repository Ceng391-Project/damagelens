import json
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from seg import DEV, augment

ROOT = Path(__file__).parent
D = ROOT / "data" / "xbd512"
OUT = ROOT / "outputs" / "xbd"
OUT.mkdir(parents=True, exist_ok=True)
EPOCHS, BS = 12, 8


def load(split):
    return (np.load(D / f"{split}_x.npy", mmap_mode="r"), np.load(D / f"{split}_y.npy", mmap_mode="r"),
            json.load(open(D / f"{split}_events.json")))


def tier3_split():
    _, _, ev = load("tier3")
    rng = np.random.default_rng(0)
    part = np.zeros(len(ev), np.int8)  # 0 train, 1 val, 2 test
    for e in set(ev):
        idx = np.array([i for i, x in enumerate(ev) if x == e]); rng.shuffle(idx)
        n = len(idx); part[idx[: int(.2 * n)]] = 2; part[idx[int(.2 * n): int(.3 * n)]] = 1
    return part


if __name__ == "__main__":
    tr = load("train"); t3 = load("tier3"); ho = load("hold")
    part = tier3_split()
    index = [("train", i) for i in range(len(tr[2]))] + [("tier3", i) for i in np.where(part == 0)[0]]
    val_index = [("hold", i) for i in range(len(ho[2]))] + [("tier3", i) for i in np.where(part == 1)[0]]
    src = {"train": tr, "tier3": t3, "hold": ho}
    freq = np.bincount(np.concatenate([np.asarray(tr[1][::20]).ravel(), np.asarray(t3[1][::20]).ravel()]), minlength=256)[:5] + 1
    w = 1 / np.sqrt(freq / freq.sum()); w = w / w.mean()
    print("pixel freq", (freq / freq.sum()).round(4), "weights", w.round(2), "train", len(index), "val", len(val_index), flush=True)

    torch.manual_seed(0); np.random.seed(0)
    model = smp.Unet("resnet18", encoder_weights="imagenet", in_channels=6, classes=5).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EPOCHS * (len(index) // BS + 1))
    wt = torch.tensor(w, dtype=torch.float32, device=DEV)
    dice = smp.losses.DiceLoss("multiclass", classes=[1, 2, 3, 4], ignore_index=255)  # 255 = un-classified building
    mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=DEV).view(1, 6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225] * 2, device=DEV).view(1, 6, 1, 1)
    prep = lambda x: (x.to(DEV).permute(0, 3, 1, 2).float() / 255 - mean) / std

    def batch(idx):
        x = np.stack([src[s][0][i] for s, i in idx]); y = np.stack([src[s][1][i] for s, i in idx])
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))

    @torch.no_grad()
    def val_score():
        model.eval(); cm = np.zeros((5, 5), np.int64)
        for i in range(0, len(val_index), BS):
            x, y = batch(val_index[i:i + BS])
            p = model(prep(x)).argmax(1).cpu().numpy(); y = y.numpy(); k = y < 5
            cm += np.bincount(y[k] * 5 + p[k], minlength=25).reshape(5, 5)
        loc_tp = cm[1:, 1:].sum(); loc_f1 = 2 * loc_tp / (2 * loc_tp + cm[0, 1:].sum() + cm[1:, 0].sum())
        f1s = []
        for c in range(1, 5):
            tp = cm[c, c]; f1s.append(2 * tp / max(2 * tp + cm[1:, c].sum() - tp + cm[c, :].sum() - tp, 1))
        dmg = len(f1s) / sum(1 / max(f, 1e-6) for f in f1s)
        return float(0.3 * loc_f1 + 0.7 * dmg), float(loc_f1), [float(f) for f in f1s]

    hist, best, t0 = [], -1, time.time()
    for ep in range(EPOCHS):
        model.train(); perm = np.random.permutation(len(index)); tot = 0
        for k in range(0, len(perm), BS):
            x, y = batch([index[j] for j in perm[k:k + BS]])
            x, y = augment(prep(x), y.to(DEV))
            out = model(x)
            loss = F.cross_entropy(out, y, weight=wt, ignore_index=255) + dice(out, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss.detach())
        score, loc, f1s = val_score()
        hist.append(dict(epoch=ep, loss=tot / (len(perm) / BS), val_xview2=score, val_loc_f1=loc, val_dmg_f1=f1s, min=round((time.time() - t0) / 60, 1)))
        print(hist[-1], flush=True)
        if score > best:
            best = score; torch.save(model.state_dict(), OUT / "xbd_unet5.pt")
    json.dump(dict(hist=hist, weights=w.tolist(), n_train=len(index), n_val=len(val_index)), open(OUT / "train_log.json", "w"), indent=2)
