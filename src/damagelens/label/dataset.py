import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer

from .. import REPO_ROOT
from ..data.pairs import Pair

CLASS_VALUE = {"damaged": 1, "destroyed": 2, "intact": 3}


def read_annotation(area_dir: Path, tile_id: str) -> dict:
    p = area_dir / "annotations" / f"{tile_id}.json"
    return json.load(open(p)) if p.exists() else dict(tile=tile_id, status="todo", polygons=[])


def rasterize(polygons, size: int) -> np.ndarray:
    # later polygons win; intact is drawn first so a damaged polygon on top of it is not hidden
    img = Image.new("L", (size, size), 0); d = ImageDraw.Draw(img)
    for poly in sorted(polygons, key=lambda p: p["cls"] != "intact"):
        if len(poly["points"]) >= 3:
            d.polygon([tuple(pt) for pt in poly["points"]], fill=CLASS_VALUE[poly["cls"]])
    return np.array(img)


def load_labeled(name: str, root: Path = REPO_ROOT / "labels", binary: bool = True) -> list[Pair]:
    # binary=True follows the KATE-CD convention: damaged or destroyed = 1, intact and background = 0
    from .prepare import ensure_tiles
    area = root / name; m = ensure_tiles(area)
    out = []
    for t in m["tiles"]:
        ann = read_annotation(area, t["id"])
        if ann.get("status") != "done":
            continue
        cls = rasterize(ann["polygons"], m["tile"])
        mask = ((cls == 1) | (cls == 2)).astype(np.uint8) if binary else cls
        pre = np.array(Image.open(area / "tiles" / f"{t['id']}_pre.png").convert("RGB"))
        post = np.array(Image.open(area / "tiles" / f"{t['id']}_post.png").convert("RGB"))
        out.append(Pair(pre, post, mask, dict(dataset=f"labels/{name}", tile=t["id"], user=ann.get("user"))))
    return out


def to_geojson(name: str, root: Path = REPO_ROOT / "labels") -> dict:
    area = root / name; m = json.load(open(area / "manifest.json"))
    to_ll = Transformer.from_crs(m["crs"], "EPSG:4326", always_xy=True)
    feats = []
    for t in m["tiles"]:
        ann = read_annotation(area, t["id"])
        for poly in ann["polygons"]:
            xs = [t["x0"] + (px + 0.5) * m["gsd"] for px, _ in poly["points"]]
            ys = [t["y0"] - (py + 0.5) * m["gsd"] for _, py in poly["points"]]
            lon, lat = to_ll.transform(xs, ys)
            ring = [[float(a), float(b)] for a, b in zip(lon, lat)]
            feats.append(dict(type="Feature", geometry=dict(type="Polygon", coordinates=[ring + ring[:1]]),
                              properties=dict(cls=poly["cls"], tile=t["id"], user=ann.get("user"), status=ann.get("status"))))
    return dict(type="FeatureCollection", features=feats)
