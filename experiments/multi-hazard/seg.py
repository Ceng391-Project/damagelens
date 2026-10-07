import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def binary_scores(pred, gt, valid=None):
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    if valid is not None:
        valid = np.asarray(valid).astype(bool)
        pred, gt = pred[valid], gt[valid]
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    tn = int(pred.size - tp - fp - fn)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9), iou=tp / max(tp + fp + fn, 1),
                accuracy=(tp + tn) / max(pred.size, 1))


def augment(x, y):
    if torch.rand(1) < 0.5: x, y = x.flip(-1), y.flip(-1)
    if torch.rand(1) < 0.5: x, y = x.flip(-2), y.flip(-2)
    k = int(torch.randint(4, (1,)))
    return torch.rot90(x, k, (-2, -1)).contiguous(), torch.rot90(y, k, (-2, -1)).contiguous()


def train_binary(Xtr, Ytr, Xva, Yva, epochs=30, bs=8, encoder="resnet18", lr=3e-4, pos_weight=2.0,
                 ignore=255, log=print, crop=None):
    """X: float32 (N,C,H,W) already normalised; Y: uint8 (N,H,W) with `ignore` for invalid px."""
    torch.manual_seed(0); np.random.seed(0)
    C = Xtr.shape[1]
    model = smp.Unet(encoder, encoder_weights="imagenet" if C == 3 else None, in_channels=C, classes=1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr_t, Ytr_t = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    best, state, hist, t0 = -1, None, [], time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            x, y = Xtr_t[idx], Ytr_t[idx]
            if crop:
                h = x.shape[-1]; a, b = np.random.randint(0, h - crop + 1, 2)
                x, y = x[..., a:a + crop, b:b + crop], y[..., a:a + crop, b:b + crop]
            x, y = augment(x.to(DEV).float(), y.to(DEV))
            valid = (y != ignore).float()[:, None]
            t = (y == 1).float()[:, None]
            out = model(x)
            bce = F.binary_cross_entropy_with_logits(out, t, pos_weight=torch.tensor(pos_weight, device=DEV), reduction="none")
            bce = (bce * valid).sum() / valid.sum().clamp(min=1)
            pr = torch.sigmoid(out) * valid
            dice = 1 - (2 * (pr * t).sum() + 1) / (pr.sum() + (t * valid).sum() + 1)
            loss = bce + dice
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        prob = predict(model, Xva)
        f1 = binary_scores(prob > 0.5, Yva == 1, Yva != ignore)["f1"]
        hist.append(dict(epoch=ep, loss=float(loss.detach()), val_f1=f1, sec=round(time.time() - t0)))
        log(hist[-1])
        if f1 > best:
            best, state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, hist


@torch.no_grad()
def predict(model, X, bs=8):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + bs]).to(DEV).float())).cpu().numpy()[:, 0])
    return np.concatenate(out)
