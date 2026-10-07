import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from rasterio.features import rasterize
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from skimage.filters import threshold_otsu

from eo import grid, s2_mosaic
from seg import binary_scores

ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "fire"
OUT.mkdir(parents=True, exist_ok=True)
NIFC = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/{}/FeatureServer/0/query"

FIRES = {
    "Palisades (LA, Oca 2025)": dict(src="nifc", svc="WFIGS_Interagency_Perimeters",
                                     where="poly_IncidentName='PALISADES' AND poly_GISAcres>20000", pre="2024-12-15/2025-01-06", post="2025-01-25/2025-02-15"),
    "Eaton (LA, Oca 2025)": dict(src="nifc", svc="WFIGS_Interagency_Perimeters",
                                 where="poly_IncidentName='Eaton' AND poly_GISAcres>10000", pre="2024-12-15/2025-01-06", post="2025-01-25/2025-02-15"),
    "Camp (Paradise, Kas 2018)": dict(src="nifc", svc="InterAgencyFirePerimeterHistory_All_Years_View",
                                      where="INCIDENT='CAMP' AND FIRE_YEAR_INT=2018 AND GIS_ACRES>100000", pre="2018-10-15/2018-11-07", post="2018-11-26/2018-12-31"),
    "Manavgat (Antalya, Tem 2021)": dict(src="effis", bbox=[31.2, 36.6, 32.3, 37.3], date="2021-07-2", pre="2021-07-05/2021-07-27", post="2021-08-10/2021-09-05"),
}


def perimeter(cfg):
    if cfg["src"] == "nifc":
        r = requests.get(NIFC.format(cfg["svc"]), params=dict(where=cfg["where"], outFields="*", outSR=4326, f="geojson"), timeout=120).json()
        return unary_union([shape(f["geometry"]) for f in r["features"]])
    b = cfg["bbox"]
    url = ("https://maps.effis.emergency.copernicus.eu/effis?service=WFS&version=1.1.0&request=GetFeature&typename=ms:modis.ba.poly"
           f"&bbox={b[1]},{b[0]},{b[3]},{b[2]},EPSG:4326&outputformat=geojson")
    feats = [f for f in requests.get(url, timeout=120).json()["features"] if f["properties"]["FIREDATE"].startswith(cfg["date"])]
    feats = sorted(feats, key=lambda f: -float(f["properties"]["AREA_HA"]))[:1]
    return unary_union([shape(f["geometry"]) for f in feats])


def nbr(b):
    a, c = np.clip(b["B08"], 0, None), np.clip(b["B12"], 0, None)
    return np.clip((a - c) / (a + c + 1e-3), -1, 1)


results = {}
fig_maps = []
for name, cfg in FIRES.items():
    poly = perimeter(cfg)
    w, s, e, n = poly.bounds; pw, ph = (e - w) * .25, (n - s) * .25
    bbox = [w - pw, s - ph, e + pw, n + ph]
    res = max((bbox[2] - bbox[0]) / 2500, 0.0002)
    cache = OUT / f"cache_{name.split()[0].lower()}.npz"
    if cache.exists():
        z = np.load(cache); pre = {"B08": z["a8"], "B12": z["a12"]}; post = {"B08": z["b8"], "B12": z["b12"]}
        pre_ids, post_ids = list(z["pi"]), list(z["qi"])
    else:
        pre, pre_ids = s2_mosaic(bbox, cfg["pre"], res, bands=("B08", "B12"), max_cloud=40)
        post, post_ids = s2_mosaic(bbox, cfg["post"], res, bands=("B08", "B12"), max_cloud=40)
        np.savez_compressed(cache, a8=pre["B08"], a12=pre["B12"], b8=post["B08"], b12=post["B12"], pi=pre_ids, qi=post_ids)
    dnbr = np.clip(nbr(pre) - nbr(post), -2, 2)
    tr, W, H = grid(bbox, res)
    gt = rasterize([mapping(poly)], out_shape=(H, W), transform=tr).astype(bool)
    valid = np.isfinite(dnbr)
    otsu = float(threshold_otsu(dnbr[valid]))
    r = {}
    for lab, t in [("dNBR>0.10 (USGS düşük şiddet)", 0.10), ("dNBR>0.27 (USGS orta-düşük)", 0.27), (f"Otsu ({otsu:.2f})", otsu)]:
        p = dnbr > t
        sc = binary_scores(p, gt, valid)
        lat = (s + n) / 2
        px_km2 = (res * 111.32) * (res * 111.32 * np.cos(np.radians(lat)))
        sc.update(area_pred_km2=float((p & valid).sum() * px_km2), area_official_km2=float(gt.sum() * px_km2))
        r[lab] = sc
    inside = dnbr[gt & valid]
    sev = {"yanmamış (<0.1)": float((inside < .1).mean()), "düşük (0.1–0.27)": float(((inside >= .1) & (inside < .27)).mean()),
           "orta-düşük (0.27–0.44)": float(((inside >= .27) & (inside < .44)).mean()), "orta-yüksek (0.44–0.66)": float(((inside >= .44) & (inside < .66)).mean()),
           "yüksek (≥0.66)": float((inside >= .66).mean())}
    results[name] = dict(scores=r, severity_inside=sev, pre_scenes=len(pre_ids), post_scenes=len(post_ids),
                         valid_fraction=float(valid.mean()), res_deg=res)
    fig_maps.append((name, nbr(pre), nbr(post), dnbr, gt, dnbr > 0.10, bbox, valid))
    print(name, {k: round(v["iou"], 3) for k, v in r.items()}, flush=True)
json.dump(results, open(OUT / "results.json", "w"), indent=2)

fig, ax = plt.subplots(len(fig_maps), 4, figsize=(18, 4.3 * len(fig_maps)))
for i, (name, a, b, d, gt, p, bb, vd) in enumerate(fig_maps):
    ext = [bb[0], bb[2], bb[1], bb[3]]
    ax[i, 0].imshow(a, cmap="RdYlGn", vmin=-.5, vmax=.8, extent=ext); ax[i, 0].set_title(f"{name}\nNBR öncesi")
    ax[i, 1].imshow(b, cmap="RdYlGn", vmin=-.5, vmax=.8, extent=ext); ax[i, 1].set_title("NBR sonrası")
    im = ax[i, 2].imshow(d, cmap="inferno", vmin=-.1, vmax=1, extent=ext); ax[i, 2].set_title("dNBR (yanma şiddeti)")
    ax[i, 2].contour(np.flipud(gt).astype(float), [.5], colors="c", linewidths=.8, extent=ext, origin="lower")
    ov = np.zeros(gt.shape + (3,)); ov[p & gt] = [0, .7, 0]; ov[p & ~gt] = [.9, .2, .2]; ov[~p & gt] = [.2, .3, .9]; ov[~vd] = [.6, .6, .6]
    ax[i, 3].imshow(ov, extent=ext); ax[i, 3].set_title("yeşil=doğru, kırmızı=yanlış alarm,\nmavi=kaçırılan, gri=bulut/veri yok", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "maps.png", dpi=70); plt.close()

names = list(results); labs0 = list(next(iter(results.values()))["scores"])
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
x = np.arange(len(names))
ax[0].bar(x - .2, [results[n]["scores"][labs0[0]]["iou"] for n in names], .4, label="dNBR>0.10")
ax[0].bar(x + .2, [list(results[n]["scores"].values())[2]["iou"] for n in names], .4, label="Otsu")
ax[0].set_xticks(x, names, rotation=15, ha="right"); ax[0].set_ylim(0, 1); ax[0].set_ylabel("IoU (resmi sınıra göre)"); ax[0].legend(); ax[0].grid(axis="y", alpha=.3)
ax[0].set_title("Yanık alanı tespiti")
ax[1].bar(x - .2, [results[n]["scores"][labs0[0]]["area_official_km2"] for n in names], .4, label="Resmi alan")
ax[1].bar(x + .2, [results[n]["scores"][labs0[0]]["area_pred_km2"] for n in names], .4, label="Uydu (dNBR>0.10)")
ax[1].set_xticks(x, names, rotation=15, ha="right"); ax[1].set_ylabel("km²"); ax[1].legend(); ax[1].grid(axis="y", alpha=.3); ax[1].set_title("Alan tahmini (tampon bölge dahil)")
sev_keys = list(next(iter(results.values()))["severity_inside"]); bottom = np.zeros(len(names))
cols = ["#cfcfcf", "#ffe08a", "#f4a259", "#d1495b", "#5c1a33"]
for k, c in zip(sev_keys, cols):
    v = np.array([results[n]["severity_inside"][k] for n in names]); ax[2].bar(x, v, bottom=bottom, label=k, color=c); bottom += v
ax[2].set_xticks(x, names, rotation=15, ha="right"); ax[2].set_ylabel("sınır içi oran"); ax[2].legend(fontsize=8); ax[2].set_title("Resmi sınır içinde yanma şiddeti dağılımı")
plt.tight_layout(); plt.savefig(OUT / "metrics.png", dpi=100); plt.close()
