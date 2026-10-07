import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).parent
SRC = ROOT.parent.parent / "data" / "xbd"
DST = ROOT.parent.parent / "data" / "xbd512"
S = 512


def dec(cell, mask=False):
    im = Image.open(io.BytesIO(cell["bytes"]))
    if mask:
        return np.array(im.resize((S, S), Image.NEAREST), dtype=np.uint8)
    return np.array(im.convert("RGB").resize((S, S), Image.BOX), dtype=np.uint8)


def prep(split, cap=None):
    files = sorted(SRC.glob(f"{split}-*.parquet"))
    names = []
    for f in files:
        names += pq.read_table(f, columns=["image_name"]).column(0).to_pylist()
    ev = [n.split("/")[-1].rsplit("_", 1)[0] for n in names]
    keep = np.ones(len(names), bool)
    if cap:
        rng = np.random.default_rng(0)
        for e in set(ev):
            idx = [i for i, x in enumerate(ev) if x == e]
            if len(idx) > cap:
                keep[rng.choice(idx, len(idx) - cap, replace=False)] = False
    n = int(keep.sum())
    DST.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(DST / f"{split}_x.npy", "w+", np.uint8, (n, S, S, 6))
    Y = np.lib.format.open_memmap(DST / f"{split}_y.npy", "w+", np.uint8, (n, S, S))
    j, gi, meta = 0, 0, []
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=16):
            for r in b.to_pylist():
                if keep[gi]:
                    X[j, ..., :3] = dec(r["t1_image"]); X[j, ..., 3:] = dec(r["t2_image"])
                    Y[j] = dec(r["t2_mask"], mask=True)
                    meta.append(ev[gi]); j += 1
                gi += 1
        print(split, f.name, j, flush=True)
    X.flush(); Y.flush()
    json.dump(meta, open(DST / f"{split}_events.json", "w"))
    print(split, Counter(meta))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        split, _, cap = arg.partition(":")
        prep(split, int(cap) if cap else None)
