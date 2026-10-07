import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from .. import DATA_DIR

EVENT_DATE = "2023-02-06"


def load_index(path: Path = DATA_DIR / "maxar_tr" / "index.json") -> list[dict]:
    return [m for m in json.load(open(path)) if m.get("visual")]


def items_near(index, lon, lat, radius_deg=0.012):
    box = (lon - radius_deg, lat - radius_deg, lon + radius_deg, lat + radius_deg)
    hit = [m for m in index if m["bbox"][0] < box[2] and m["bbox"][2] > box[0] and m["bbox"][1] < box[3] and m["bbox"][3] > box[1]]
    by_date = lambda c: {d: [m for m in c if m["dt"][:10] == d] for d in sorted({m["dt"][:10] for m in c})}
    return by_date([m for m in hit if m["dt"] < EVENT_DATE]), by_date([m for m in hit if m["dt"] >= EVENT_DATE])


def mosaic(items, lon, lat, side_m, gsd, crs="EPSG:32637") -> np.ndarray:
    cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    n = int(side_m / gsd); tr = from_origin(cx - side_m / 2, cy + side_m / 2, gsd, gsd)
    out = np.zeros((n, n, 3), np.uint8)
    for m in items:
        with rasterio.open(m["visual"]) as src, WarpedVRT(src, crs=crs, transform=tr, width=n, height=n, resampling=Resampling.average) as v:
            a = v.read([1, 2, 3]).transpose(1, 2, 0)
        fill = (out.sum(-1) == 0) & (a.sum(-1) > 0); out[fill] = a[fill]
    return out


def snow_cloud_fraction(rgb: np.ndarray) -> float:
    return float(((rgb.min(-1) > 190) & (rgb.max(-1) - rgb.min(-1) < 30)).mean())


def pick_dates(pre_by_date, post_by_date, lon, lat, side_m):
    # pre: latest full-coverage date (closest to the event); post: earliest full-coverage date with <5 % snow/cloud,
    # otherwise the least snowy one
    def check(items):
        x = mosaic(items, lon, lat, side_m, 4.0)
        return (x.sum(-1) > 0).mean() > 0.9, snow_cloud_fraction(x)
    pre = next(d for d in reversed(pre_by_date) if check(pre_by_date[d])[0])
    post_ok = {d: c[1] for d in post_by_date if (c := check(post_by_date[d]))[0]}
    clean = [d for d, snow in post_ok.items() if snow < 0.05]
    return pre, clean[0] if clean else min(post_ok, key=post_ok.get)
