import argparse
import json
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from common import OUT, load_kate, load_xbd, scores

p = argparse.ArgumentParser()
p.add_argument("--train", choices=["kate", "xbd"], required=True)
p.add_argument("--init", default=None)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--name", required=True)
args = p.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def to_tensor(data):
    x = np.stack([np.concatenate([a, b], -1) for a, b, *_ in data]).transpose(0, 3, 1, 2)
    y = np.stack([m for _, _, m, *_ in data])[:, None]
    return torch.from_numpy(x), torch.from_numpy(y)


train = load_kate("train") if args.train == "kate" else load_xbd()
Xtr, Ytr = to_tensor(train)
Xva, Yva = to_tensor(load_kate("validation"))
Xte, Yte = to_tensor(load_kate("test"))
print(f"train tiles {len(Xtr)}, damaged px {Ytr.float().mean():.4f}")

model = smp.Unet("resnet18", encoder_weights="imagenet", in_channels=6, classes=1).to(dev)
if args.init:
    model.load_state_dict(torch.load(OUT / f"{args.init}.pt", map_location=dev))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4 if not args.init else 1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
dice = smp.losses.DiceLoss("binary")
mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=dev).view(1, 6, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225] * 2, device=dev).view(1, 6, 1, 1)


def prep(x):
    return (x.to(dev).float() / 255 - mean) / std


@torch.no_grad()
def predict(X):
    model.eval()
    return torch.cat([torch.sigmoid(model(prep(X[i:i + 8]))).cpu() for i in range(0, len(X), 8)])


def best_threshold(prob, Y):
    f1 = {t: scores(prob > t, Y)["f1"] for t in np.arange(0.1, 0.91, 0.1)}
    t = max(f1, key=f1.get)
    return float(t), float(f1[t])


log, best = [], (-1, None)
t0 = time.time()
for ep in range(args.epochs):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), 8):
        idx = perm[i:i + 8]
        x, y = prep(Xtr[idx]), Ytr[idx].to(dev).float()
        if torch.rand(1) < 0.5: x, y = x.flip(-1), y.flip(-1)
        if torch.rand(1) < 0.5: x, y = x.flip(-2), y.flip(-2)
        k = int(torch.randint(4, (1,)))
        x, y = torch.rot90(x, k, (-2, -1)).contiguous(), torch.rot90(y, k, (-2, -1)).contiguous()
        out = model(x)
        loss = F.binary_cross_entropy_with_logits(out, y, pos_weight=torch.tensor(5.0, device=dev)) + dice(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    thr, f1 = best_threshold(predict(Xva), Yva)
    log.append(dict(epoch=ep, loss=float(loss), val_f1=f1, thr=thr, sec=round(time.time() - t0)))
    print(log[-1], flush=True)
    if f1 > best[0]:
        best = (f1, thr)
        torch.save(model.state_dict(), OUT / f"{args.name}.pt")

model.load_state_dict(torch.load(OUT / f"{args.name}.pt", map_location=dev))
test = {k: float(v) for k, v in scores(predict(Xte) > best[1], Yte).items()}
print("TEST", args.name, test)
json.dump(dict(args=vars(args), best_val_f1=best[0], thr=best[1], test=test, log=log,
               train_minutes=round((time.time() - t0) / 60, 1)),
          open(OUT / f"{args.name}.json", "w"), indent=2)
