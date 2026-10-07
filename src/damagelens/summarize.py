import argparse
import json
from pathlib import Path

import numpy as np

from .align import register_phase
from .data import tiles
from .models import load_model, predict


def damage_map(model, pre: np.ndarray, post: np.ndarray, tile: int = 512, register: bool = True, max_shift: float = 40):
    h, w = post.shape[:2]
    prob = np.zeros((h, w), np.float32); shifts = []
    for y, x in tiles(h, w, tile):
        a, b = pre[y:y + tile, x:x + tile], post[y:y + tile, x:x + tile]
        if register:
            a, sh, ok = register_phase(b, a, max_shift)
            shifts.append(float(np.hypot(*sh)) if ok else float("nan"))
        prob[y:y + tile, x:x + tile] = predict(model, a, b)
    return prob, shifts


def grid_summary(mask: np.ndarray, cell_px: int) -> np.ndarray:
    k0, k1 = mask.shape[0] // cell_px, mask.shape[1] // cell_px
    return mask[: k0 * cell_px, : k1 * cell_px].reshape(k0, cell_px, k1, cell_px).mean((1, 3))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Damage grid for a pre/post scene stored as .npy (H, W, 3) uint8")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pre", type=Path, required=True)
    ap.add_argument("--post", type=Path, required=True)
    ap.add_argument("--gsd", type=float, default=0.5, help="metres per pixel")
    ap.add_argument("--cell-m", type=float, default=48)
    ap.add_argument("--out", type=Path, default=Path("damage_grid.npy"))
    a = ap.parse_args(argv)
    model, meta = load_model(a.model)
    prob, shifts = damage_map(model, np.load(a.pre), np.load(a.post))
    g = grid_summary(prob > meta.get("thr", 0.5), int(round(a.cell_m / a.gsd)))
    np.save(a.out, g)
    print(json.dumps(dict(damaged_frac=float((prob > meta.get("thr", 0.5)).mean()), cells=list(g.shape),
                          median_shift_px=float(np.nanmedian(shifts)) if shifts else None), indent=1))


if __name__ == "__main__":
    main()
