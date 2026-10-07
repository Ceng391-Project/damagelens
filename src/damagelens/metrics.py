import numpy as np
from scipy import ndimage


def binary_scores(pred, gt, valid=None) -> dict:
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if valid is not None:
        v = np.asarray(valid, bool); pred, gt = pred[v], gt[v]
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9), iou=tp / max(tp + fp + fn, 1))


def best_threshold(prob, gt, grid=np.arange(0.1, 0.91, 0.1)) -> tuple[float, float]:
    f = {float(t): binary_scores(prob > t, gt)["f1"] for t in grid}
    t = max(f, key=f.get)
    return t, f[t]


def xview2_scores(cm: np.ndarray) -> dict:
    # cm rows = truth, cols = prediction; damage F1 is the harmonic mean over the 4 damage classes (xView2 definition)
    tp = cm[1:, 1:].sum(); loc = 2 * tp / max(2 * tp + cm[0, 1:].sum() + cm[1:, 0].sum(), 1)
    f1 = []
    for c in range(1, 5):
        t = cm[c, c]
        f1.append(2 * t / (2 * t + (cm[1:, c].sum() - t) + (cm[c, 1:].sum() - t)) if cm[c, 1:].sum() > 0 else np.nan)
    v = [f for f in f1 if not np.isnan(f)]
    dmg = len(v) / sum(1 / max(f, 1e-6) for f in v) if v else float("nan")
    return dict(loc_f1=float(loc), dmg_f1=float(dmg), xview2=float(0.3 * loc + 0.7 * dmg),
                dmg_f1_per_class=[None if np.isnan(f) else float(f) for f in f1])


def building_confusion(gt: np.ndarray, pred: np.ndarray, min_cover: float = 0.2) -> np.ndarray:
    # column 0 = "missed": less than min_cover of the building was predicted as any building class
    out = np.zeros((5, 5), np.int64)
    lab, n = ndimage.label((gt >= 1) & (gt <= 4))
    for c in range(1, n + 1):
        reg = lab == c
        g = np.bincount(gt[reg], minlength=5)[1:5].argmax() + 1
        p = pred[reg]; p = p[(p > 0) & (p < 5)]
        q = np.bincount(p, minlength=5)[1:5].argmax() + 1 if len(p) > min_cover * reg.sum() else 0
        out[g, q] += 1
    return out
