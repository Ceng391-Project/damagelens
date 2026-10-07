import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

CAT = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
S2_BAD_SCL = {0, 1, 3, 8, 9, 10, 11}


def grid(bbox, res):
    w, s, e, n = bbox
    return from_origin(w, n, res, res), int(round((e - w) / res)), int(round((n - s) / res))


def read_to_grid(href, bbox, res, resampling=Resampling.average, dtype="float32", nodata=None):
    tr, W, H = grid(bbox, res)
    kw = {}
    with rasterio.open(href) as src:
        if src.crs and src.crs.is_projected:
            target_m, native = res * 111000 * 0.75, abs(src.res[0])
            lv = [i for i, f in enumerate(src.overviews(1)) if native * f <= target_m]
            if lv:
                kw["overview_level"] = lv[-1]
    with rasterio.open(href, **kw) as src:
        with WarpedVRT(src, crs="EPSG:4326", transform=tr, width=W, height=H, resampling=resampling,
                       src_nodata=nodata if nodata is not None else src.nodata, nodata=np.nan if "float" in dtype else 0) as v:
            return v.read(1, out_dtype=dtype)


def s2_mosaic(bbox, date, res, bands=("B04", "B08"), max_cloud=60):
    items = list(CAT.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=date,
                            query={"eo:cloud_cover": {"lt": max_cloud}}).items())
    tr, W, H = grid(bbox, res)
    out = {b: np.full((H, W), np.nan, np.float32) for b in bands}
    for it in sorted(items, key=lambda i: i.properties["eo:cloud_cover"]):
        scl = read_to_grid(it.assets["SCL"].href, bbox, res, Resampling.nearest, "float32")
        ok = np.isfinite(scl) & ~np.isin(scl, list(S2_BAD_SCL)) & (scl > 0)
        need = ok & np.isnan(out[bands[0]])
        if not need.any():
            continue
        for b in bands:
            a = read_to_grid(it.assets[b].href, bbox, res)
            off = -1000 if str(it.properties.get("s2:processing_baseline", "0")) >= "04.00" else 0
            out[b][need] = (a[need] + off) / 10000
    return out, [i.id for i in items]


def worldcover(bbox, res, year=2021):
    items = list(CAT.search(collections=["esa-worldcover"], bbox=bbox,
                            query={"esa_worldcover:product_version": {"eq": "2.0.0" if year == 2021 else "1.0.0"}}).items())
    tr, W, H = grid(bbox, res)
    out = np.zeros((H, W), np.uint8)
    for it in items:
        a = read_to_grid(it.assets["map"].href, bbox, res, Resampling.mode, "uint8", nodata=0)
        out[out == 0] = a[out == 0]
    return out
