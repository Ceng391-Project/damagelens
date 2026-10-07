from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .. import DATA_DIR
from .pairs import Pair, decode_image, decode_mask

SPLITS = ("train", "validation", "test")


def load_kate(split: str, root: Path = DATA_DIR / "kate-cd") -> list[Pair]:
    rows = pq.read_table(root / f"{split}.parquet").to_pylist()
    return [Pair(decode_image(r["pre_image"]), decode_image(r["post_image"]),
                 (decode_mask(r["label"]) > 0).astype(np.uint8), dict(dataset="kate-cd", split=split, index=i))
            for i, r in enumerate(rows)]
