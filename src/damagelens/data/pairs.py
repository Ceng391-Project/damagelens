import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class Pair:
    pre: np.ndarray            # uint8 (H, W, 3)
    post: np.ndarray           # uint8 (H, W, 3)
    mask: np.ndarray | None    # uint8 (H, W): binary {0,1} or xBD classes {0..4, 255 = ignore}
    meta: dict = field(default_factory=dict)


def decode_image(cell) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def decode_mask(cell) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(cell["bytes"])))
