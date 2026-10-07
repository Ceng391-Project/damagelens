import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def _img(cell):
    return np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def _mask(cell):
    return np.array(Image.open(io.BytesIO(cell["bytes"])))


def load_kate(split):
    rows = pq.read_table(DATA / "kate-cd" / f"{split}.parquet").to_pylist()
    return [(_img(r["pre_image"]), _img(r["post_image"]), (_mask(r["label"]) > 0).astype(np.uint8)) for r in rows]


def load_xbd(min_damage=3, size=512):
    # xBD t2_mask: 0 bg, 1 no-damage, 2 minor, 3 major, 4 destroyed
    out = []
    # first 4 of the 17 xBD train shards, as in the feasibility run (download_data.sh xbd)
    for f in sorted((ROOT.parent / "disaster-eval" / "data" / "xbd").glob("train-0000[0-3]-of-00017.parquet")):
        for r in pq.read_table(f).to_pylist():
            pre, post = _img(r["t1_image"]), _img(r["t2_image"])
            m2 = _mask(r["t2_mask"])
            dmg = (m2 >= min_damage).astype(np.uint8)
            for y in range(0, pre.shape[0], size):
                for x in range(0, pre.shape[1], size):
                    s = np.s_[y:y + size, x:x + size]
                    if m2[s].any():
                        out.append((pre[s], post[s], dmg[s], r["image_name"]))
    return out


def scores(pred, gt):
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9), iou=tp / max(tp + fp + fn, 1))
