import json
from pathlib import Path

import numpy as np

from .. import DATA_DIR

CLASSES = ["background", "no-damage", "minor", "major", "destroyed"]  # mask value 255 = un-classified building
HAZARD = {
    "mexico-earthquake": "earthquake", "palu-tsunami": "tsunami", "sunda-tsunami": "tsunami",
    "midwest-flooding": "flood", "nepal-flooding": "flood",
    "hurricane-harvey": "hurricane", "hurricane-florence": "hurricane", "hurricane-michael": "hurricane", "hurricane-matthew": "hurricane",
    "joplin-tornado": "tornado", "moore-tornado": "tornado", "tuscaloosa-tornado": "tornado",
    "socal-fire": "fire", "santa-rosa-wildfire": "fire", "woolsey-fire": "fire", "portugal-wildfire": "fire", "pinery-bushfire": "fire",
    "guatemala-volcano": "volcano", "lower-puna-volcano": "volcano",
}


def load_xbd512(split: str, root: Path = DATA_DIR / "xbd512"):
    # written by experiments/multi-hazard/xbd_prep.py; x = pre RGB + post RGB (N, 512, 512, 6)
    x = np.load(root / f"{split}_x.npy", mmap_mode="r")
    y = np.load(root / f"{split}_y.npy", mmap_mode="r")
    return x, y, json.load(open(root / f"{split}_events.json"))
