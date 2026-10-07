import argparse
import json
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from . import RUNS_DIR
from .augment import flip_rotate, misregister
from .data import load_kate
from .metrics import best_threshold, binary_scores
from .models import build_unet, normalize, predict, save_model
from .models.unet import device


def stack(pairs):
    x = np.stack([np.concatenate([p.pre, p.post], -1) for p in pairs])
    y = np.stack([p.mask for p in pairs])
    return x, y


def negatives_from_pre(pairs, n: int, seed: int = 0):
    # KATE-CD has damage in every tile; (pre, jittered pre) pairs give the model no-change scenes to learn from
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.choice(len(pairs), min(n, len(pairs)), replace=False):
        a = pairs[i].pre.astype(np.float32)
        b = np.clip(a * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20, 3), 0, 255).astype(np.uint8)
        out.append(np.concatenate([pairs[i].pre, b], -1))
    return np.stack(out), np.zeros((len(out),) + pairs[0].mask.shape, np.uint8)


def train_binary(train, val, epochs=30, batch=8, lr=3e-4, shift_px=0, negatives=0, pos_weight=5.0, seed=0, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = device()
    xtr, ytr = stack(train)
    if negatives:
        xn, yn = negatives_from_pre(train, negatives, seed)
        xtr, ytr = np.concatenate([xtr, xn]), np.concatenate([ytr, yn])
    xtr, ytr = torch.from_numpy(xtr), torch.from_numpy(ytr)[:, None]
    model = build_unet(1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dice = smp.losses.DiceLoss("binary")
    vpre = np.stack([p.pre for p in val]); vpost = np.stack([p.post for p in val]); vy = np.stack([p.mask for p in val])
    best, state, hist, t0 = (-1.0, 0.5), None, [], time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            x, y = normalize(xtr[idx].to(dev)), ytr[idx].to(dev).float()
            x, y = flip_rotate(x, y); x = misregister(x, shift_px)
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out, y, pos_weight=torch.tensor(pos_weight, device=dev)) + dice(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        thr, f1 = best_threshold(predict(model, vpre, vpost), vy)
        hist.append(dict(epoch=ep, loss=float(loss.detach()), val_f1=f1, thr=thr, sec=round(time.time() - t0)))
        log(hist[-1])
        if f1 > best[0]:
            best, state = (f1, thr), {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, dict(val_f1=best[0], thr=best[1], hist=hist)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the 6-channel U-Net on KATE-CD")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--shift-px", type=int, default=0, help="random pre-image misregistration during training")
    ap.add_argument("--negatives", type=int, default=0, help="number of synthetic undamaged (pre, pre) tiles to add")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    model, info = train_binary(load_kate("train"), load_kate("validation"), a.epochs, shift_px=a.shift_px, negatives=a.negatives, seed=a.seed)
    test = load_kate("test")
    prob = predict(model, np.stack([p.pre for p in test]), np.stack([p.post for p in test]))
    info["test"] = {k: float(v) for k, v in binary_scores(prob > info["thr"], np.stack([p.mask for p in test])).items()}
    out = RUNS_DIR / a.name / "model.pt"
    save_model(model, out, classes=1, encoder="resnet18", thr=info["thr"], args=vars(a), val_f1=info["val_f1"], test=info["test"], hist=info["hist"])
    print(json.dumps(dict(name=a.name, val_f1=info["val_f1"], thr=info["thr"], test=info["test"]), indent=1))


if __name__ == "__main__":
    main()
