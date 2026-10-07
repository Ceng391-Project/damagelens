import argparse
import json
from pathlib import Path

import numpy as np

from .align import apply_shift, register_phase
from .baselines import BASELINES
from .data import load_kate
from .metrics import binary_scores, best_threshold
from .models import load_model, predict


def misregister_pre(pairs, px: float):
    return [apply_shift(p.pre, np.array([px, 0.6 * px])) for p in pairs]


def evaluate_model(path, split="test", shifts=(0,), register=False):
    model, meta = load_model(path)
    thr = meta.get("thr", 0.5)
    data = load_kate(split); gt = np.stack([p.mask for p in data]); post = np.stack([p.post for p in data])
    out = {}
    for px in shifts:
        pre = misregister_pre(data, px) if px else [p.pre for p in data]
        if register:
            pre = [register_phase(b, a)[0] for a, b in zip(pre, post)]
        out[px] = {k: float(v) for k, v in binary_scores(predict(model, np.stack(pre), post) > thr, gt).items()}
    return out


def evaluate_baselines(names=tuple(BASELINES), split="test"):
    val, test = load_kate("validation"), load_kate(split)
    out = {}
    for n in names:
        fn = BASELINES[n]
        score = lambda d: np.stack([fn(p.pre, p.post) for p in d])
        sv, st = score(val), score(test)
        lo, hi = np.percentile(sv, [50, 99.5])
        thr, _ = best_threshold(sv, np.stack([p.mask for p in val]), grid=np.linspace(lo, hi, 15))
        out[n] = {k: float(v) for k, v in binary_scores(st > thr, np.stack([p.mask for p in test])).items()} | {"thr": float(thr)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a model and/or the classical baselines on KATE-CD")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--baselines", nargs="*", default=None, help=f"subset of {list(BASELINES)}; empty = all")
    ap.add_argument("--shifts", type=float, nargs="*", default=[0])
    ap.add_argument("--register", action="store_true", help="phase-correlation registration before inference")
    a = ap.parse_args(argv)
    res = {}
    if a.model:
        res["model"] = evaluate_model(a.model, shifts=a.shifts, register=a.register)
    if a.baselines is not None:
        res["baselines"] = evaluate_baselines(tuple(a.baselines) or tuple(BASELINES))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
