import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer

from .. import REPO_ROOT
from ..align import register_phase
from ..data import tiles
from ..data.maxar import items_near, load_index, mosaic, pick_dates, snow_cloud_fraction

TILE = 512
CRS = "EPSG:32637"


def prepare_area(name, lon, lat, side_m=1536, gsd=0.5, pre_date=None, post_date=None, root=REPO_ROOT / "labels", write_manifest=True):
    out = root / name; (out / "tiles").mkdir(parents=True, exist_ok=True); (out / "annotations").mkdir(exist_ok=True)
    pre_d, post_d = items_near(load_index(), lon, lat)
    if not (pre_date and post_date):
        pre_date, post_date = pick_dates(pre_d, post_d, lon, lat, side_m)
    pre = mosaic(pre_d[pre_date], lon, lat, side_m, gsd, CRS)
    post = mosaic(post_d[post_date], lon, lat, side_m, gsd, CRS)
    cx, cy = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    x_left, y_top = cx - side_m / 2, cy + side_m / 2
    entries = []
    for k, (r, c) in enumerate(tiles(*post.shape[:2], TILE)):
        a, b = pre[r:r + TILE, c:c + TILE], post[r:r + TILE, c:c + TILE]
        a, sh, ok = register_phase(b, a)
        tid = f"{name}_{k:03d}"
        Image.fromarray(a).save(out / "tiles" / f"{tid}_pre.png")
        Image.fromarray(b).save(out / "tiles" / f"{tid}_post.png")
        entries.append(dict(id=tid, row=r, col=c, x0=x_left + c * gsd, y0=y_top - r * gsd,
                            pre_shift_px=[float(sh[0]), float(sh[1])], registered=bool(ok),
                            snow_cloud=snow_cloud_fraction(b)))
    manifest = dict(name=name, lon=lon, lat=lat, side_m=side_m, gsd=gsd, crs=CRS, tile=TILE,
                    pre_date=pre_date, post_date=post_date, created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    source="Maxar Open Data, Kahramanmaras-turkey-earthquake-23 (CC BY-NC 4.0)", tiles=entries)
    if write_manifest:
        json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    return out, manifest


def ensure_tiles(area_dir: Path):
    # tiles are not in git (Maxar licence, size); the committed manifest pins the dates so they come out identical
    m = json.load(open(area_dir / "manifest.json"))
    if all((area_dir / "tiles" / f"{t['id']}_post.png").exists() for t in m["tiles"]):
        return m
    prepare_area(m["name"], m["lon"], m["lat"], m["side_m"], m["gsd"], m["pre_date"], m["post_date"], area_dir.parent, write_manifest=False)
    return m
