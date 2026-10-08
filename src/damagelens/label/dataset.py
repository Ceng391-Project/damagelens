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
            ring = [[float(a), float(b)] for a, b in zip(lon, lat, strict=False)]
            feats.append(dict(type="Feature", geometry=dict(type="Polygon", coordinates=[ring + ring[:1]]),
                              properties=dict(cls=poly["cls"], tile=t["id"], user=ann.get("user"), status=ann.get("status"))))
    return dict(type="FeatureCollection", features=feats)


STATUSES = ("todo", "done", "skip")


def validate_area(area_dir: Path, min_area_px: float = 4.0) -> list[str]:
    errors = []
    m = json.load(open(area_dir / "manifest.json"))
    ids = [t["id"] for t in m["tiles"]]
    if len(ids) != len(set(ids)):
        errors.append("manifest: duplicate tile ids")
    size = m["tile"]
    for f in sorted((area_dir / "annotations").glob("*.json")):
        where = f"{area_dir.name}/{f.name}"
        try:
            ann = json.load(open(f))
        except json.JSONDecodeError as e:
            errors.append(f"{where}: invalid JSON ({e})"); continue
        if f.stem not in ids or ann.get("tile") != f.stem:
            errors.append(f"{where}: tile id not in manifest or does not match the file name")
        if ann.get("status") not in STATUSES:
            errors.append(f"{where}: status must be one of {STATUSES}")
        if ann.get("status") == "done" and not ann.get("user"):
            errors.append(f"{where}: done tiles need a user name")
        for k, poly in enumerate(ann.get("polygons", [])):
            pts = poly.get("points", [])
            if poly.get("cls") not in CLASS_VALUE:
                errors.append(f"{where}: polygon {k} has unknown class {poly.get('cls')!r}")
            if len(pts) < 3:
                errors.append(f"{where}: polygon {k} has fewer than 3 points"); continue
            if any(not (-1 <= x <= size + 1 and -1 <= y <= size + 1) for x, y in pts):
                errors.append(f"{where}: polygon {k} leaves the {size}px tile")
            area = 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1], strict=False)))
            if area < min_area_px:
                errors.append(f"{where}: polygon {k} is degenerate ({area:.1f} px²)")
    return errors
