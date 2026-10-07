import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meteostat as ms
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from sklearn.metrics import roc_auc_score, roc_curve

from eo import CAT, grid, read_to_grid

ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "heat"
OUT.mkdir(parents=True, exist_ok=True)

CITIES = {  # 2023 summer heatwaves
    "Phoenix": (33.43, -112.01), "Las Vegas": (36.08, -115.15), "El Paso": (31.81, -106.38),
    "Sevilla": (37.42, -5.90), "Roma": (41.80, 12.24), "Palermo": (38.18, 13.10), "Atina": (37.94, 23.94),
    "Antalya": (36.90, 30.79), "Adana": (36.98, 35.28), "Pekin": (40.08, 116.58),
}
YEARS = {"calib": 2022, "test": 2023}
SINU = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"


def lst_series(lat, lon, year):
    items = list(CAT.search(collections=["modis-11A1-061"], bbox=[lon - .5, lat - .5, lon + .5, lat + .5],
                            datetime=f"{year}-06-01/{year}-08-31").items())
    x, y = Transformer.from_crs("EPSG:4326", SINU, always_xy=True).transform(lon, lat)

    def one(it):
        if it.properties.get("platform", "terra").lower() != "terra":
            return None
        try:
            with rasterio.open(it.assets["LST_Day_1km"].href) as src:
                b = src.bounds
                if not (b.left <= x < b.right and b.bottom < y <= b.top):
                    return None
                v = next(src.sample([(x, y)]))[0]
            day = (it.datetime or pd.Timestamp(it.properties["start_datetime"])).date()
            return day, (v * 0.02 - 273.15) if v > 0 else np.nan
        except Exception:
            return None
    with ThreadPoolExecutor(16) as ex:
        res = [r for r in ex.map(one, items) if r]
    out = {}
    for d, v in res:  # a point near a tile edge is returned by two tiles; keep the valid one
        if d not in out or np.isnan(out[d]):
            out[d] = v
    return pd.Series(out).sort_index()


def station_tmax(lat, lon, start, end):
    st = ms.stations.nearby(ms.Point(lat, lon), limit=1)
    df = ms.daily(st.index[0], start, end).fetch()
    return df["tmax"].astype(float), st.iloc[0]["name"]


cache = OUT / "series.json"
if cache.exists():
    data = json.load(open(cache))
    for c, d in data.items():  # refill cities whose series came back empty
        if not any(v is not None for v in d["lst_2023"].values()):
            la, lo = CITIES[c]
            for yr in YEARS.values():
                d[f"lst_{yr}"] = {str(k): (None if np.isnan(v) else float(v)) for k, v in lst_series(la, lo, yr).items()}
            print("refilled", c, sum(v is not None for v in d["lst_2023"].values()), flush=True)
    json.dump(data, open(cache, "w"))
else:
    data = {}
    for c, (la, lo) in CITIES.items():
        tmax, sname = station_tmax(la, lo, date(2013, 6, 1), date(2023, 8, 31))
        d = {"station": sname, "tmax": {str(k.date()): v for k, v in tmax.items() if pd.notna(v)}}
        for tag, yr in YEARS.items():
            s = lst_series(la, lo, yr)
            d[f"lst_{yr}"] = {str(k): (None if np.isnan(v) else float(v)) for k, v in s.items()}
        data[c] = d
        print(c, sname, len(d["tmax"]), sum(v is not None for v in d["lst_2023"].values()), flush=True)
    json.dump(data, open(cache, "w"))

rows, per_city = [], {}
for c, d in data.items():
    tmax = pd.Series(d["tmax"]); tmax.index = pd.to_datetime(tmax.index)
    jja = tmax[tmax.index.month.isin([6, 7, 8]) & (tmax.index.year <= 2022)]
    p90 = float(jja.quantile(0.90))
    frames = {}
    for yr in YEARS.values():
        lst = pd.Series(d[f"lst_{yr}"], dtype=float); lst.index = pd.to_datetime(lst.index)
        frames[yr] = pd.DataFrame({"lst": lst, "tmax": tmax}).dropna()
    cal, te = frames[YEARS["calib"]], frames[YEARS["test"]]
    a, b = np.polyfit(cal.lst, cal.tmax, 1)
    te = te.assign(pred=a * te.lst + b, heat=(te.tmax >= p90).astype(int), city=c,
                   z=(te.lst - cal.lst.mean()) / cal.lst.std())
    rate = float((cal.tmax >= p90).mean())
    lst_thr = float(cal.lst.quantile(1 - rate)) if rate > 0 else float(cal.lst.max())
    te = te.assign(qheat=te.lst >= lst_thr)
    qp = te.qheat; qtp = int((qp & (te.heat == 1)).sum()); qfp = int((qp & (te.heat == 0)).sum()); qfn = int((~qp & (te.heat == 1)).sum())
    pred_heat = te.pred >= p90
    tp = int((pred_heat & (te.heat == 1)).sum()); fp = int((pred_heat & (te.heat == 0)).sum()); fn = int((~pred_heat & (te.heat == 1)).sum())
    per_city[c] = dict(station=d["station"], p90_tmax=p90, n_days=int(len(te)), heat_days=int(te.heat.sum()),
                       pearson_r=float(np.corrcoef(te.lst, te.tmax)[0, 1]), rmse_c=float(np.sqrt(((te.pred - te.tmax) ** 2).mean())),
                       mae_c=float((te.pred - te.tmax).abs().mean()), lst_minus_tmax_mean=float((te.lst - te.tmax).mean()),
                       precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                       q_lst_threshold_c=lst_thr, q_precision=qtp / max(qtp + qfp, 1), q_recall=qtp / max(qtp + qfn, 1), q_f1=2 * qtp / max(2 * qtp + qfp + qfn, 1),
                       auc=float(roc_auc_score(te.heat, te.lst)) if 0 < te.heat.sum() < len(te) else None,
                       clear_sky_fraction=float(len(te) / 92))
    rows.append(te)
allte = pd.concat(rows)
pooled_auc = float(roc_auc_score(allte.heat, allte.z))
summary = dict(test_year=YEARS["test"], calib_year=YEARS["calib"], pooled_auc_zscore=pooled_auc,
               mean_r=float(np.mean([v["pearson_r"] for v in per_city.values()])),
               mean_rmse=float(np.mean([v["rmse_c"] for v in per_city.values()])), per_city=per_city)
P = np.concatenate([(r.pred >= per_city[r.city.iloc[0]]["p90_tmax"]).values for r in rows]); Y = allte.heat.values.astype(bool)
summary["pooled_f1"] = float(2 * (P & Y).sum() / max(2 * (P & Y).sum() + (P & ~Y).sum() + (~P & Y).sum(), 1))
summary["pooled_precision"] = float((P & Y).sum() / max(P.sum(), 1)); summary["pooled_recall"] = float((P & Y).sum() / max(Y.sum(), 1))
Q = allte.qheat.values.astype(bool)
summary["pooled_q_f1"] = float(2 * (Q & Y).sum() / max(2 * (Q & Y).sum() + (Q & ~Y).sum() + (~Q & Y).sum(), 1))
summary["pooled_q_precision"] = float((Q & Y).sum() / max(Q.sum(), 1)); summary["pooled_q_recall"] = float((Q & Y).sum() / max(Y.sum(), 1))
json.dump(summary, open(OUT / "results.json", "w"), indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "per_city"}, indent=1))
print(pd.DataFrame(per_city).T[["pearson_r", "rmse_c", "f1", "q_f1", "auc", "heat_days", "n_days"]].round(2))

# charts
fig, axs = plt.subplots(2, 5, figsize=(22, 8), sharey=False)
for ax, (c, te) in zip(axs.ravel(), [(r.city.iloc[0], r) for r in rows]):
    ax.plot(te.index, te.tmax, "k-", lw=1, label="İstasyon Tmax")
    ax.plot(te.index, te.pred, "o", ms=3, color="#e07b39", label="LST→Tmax tahmini")
    ax.axhline(per_city[c]["p90_tmax"], color="r", ls="--", lw=.8, label="aşırı sıcak eşiği (P90)")
    ax.set_title(f"{c} r={per_city[c]['pearson_r']:.2f} F1(yüzdelik)={per_city[c]['q_f1']:.2f}", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=7); ax.grid(alpha=.3)
axs[0, 0].legend(fontsize=7)
plt.suptitle("Yaz 2023: uydu (MODIS LST, 2022 ile kalibre) vs istasyon günlük maksimum sıcaklığı")
plt.tight_layout(); plt.savefig(OUT / "timeseries.png", dpi=85); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ax[0].scatter(allte.lst, allte.tmax, s=5, c=allte.heat, cmap="coolwarm")
ax[0].set_xlabel("MODIS LST gündüz (°C)"); ax[0].set_ylabel("İstasyon Tmax (°C)"); ax[0].set_title("Yüzey sıcaklığı vs hava sıcaklığı (kırmızı = aşırı sıcak günü)"); ax[0].grid(alpha=.3)
fpr, tpr, _ = roc_curve(allte.heat, allte.z)
ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], "k--", lw=.7); ax[1].set_title(f"Aşırı sıcak günü tespiti ROC (havuz AUC={pooled_auc:.2f})"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].grid(alpha=.3)
cs = list(per_city); x = np.arange(len(cs))
ax[2].bar(x - .27, [per_city[c]["pearson_r"] for c in cs], .27, label="Pearson r (LST vs Tmax)")
ax[2].bar(x, [per_city[c]["f1"] for c in cs], .27, label="F1 — doğrusal kalibrasyon")
ax[2].bar(x + .27, [per_city[c]["q_f1"] for c in cs], .27, label="F1 — yüzdelik eşleme")
ax[2].set_xticks(x, cs, rotation=40, ha="right"); ax[2].set_ylim(0, 1); ax[2].legend(); ax[2].grid(axis="y", alpha=.3); ax[2].set_title("Şehir bazında performans")
plt.tight_layout(); plt.savefig(OUT / "metrics.png", dpi=100); plt.close()

# anomaly map: mid-July 2023 8-day LST vs 2018–2022 same composite
BBOX, RES = [-10, 34, 45, 46], 0.05


def comp(year):
    items = list(CAT.search(collections=["modis-11A2-061"], bbox=BBOX, datetime=f"{year}-07-12/{year}-07-19").items())
    items = [i for i in items if i.properties.get("platform", "terra").lower() == "terra"]
    tr, W, H = grid(BBOX, RES); out = np.full((H, W), np.nan, np.float32)
    for it in items:
        a = read_to_grid(it.assets["LST_Day_1km"].href, BBOX, RES, Resampling.average, "float32", nodata=0)
        m = np.isfinite(a) & (a > 0) & np.isnan(out); out[m] = a[m] * 0.02 - 273.15
    return out


mcache = OUT / "anomaly.npz"
if mcache.exists():
    z = np.load(mcache); y23, base = z["y23"], z["base"]
else:
    y23 = comp(2023); base = np.nanmean([comp(y) for y in range(2018, 2023)], 0)
    np.savez_compressed(mcache, y23=y23, base=base)
fig, ax = plt.subplots(figsize=(15, 4.8))
im = ax.imshow(y23 - base, cmap="RdBu_r", vmin=-10, vmax=10, extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]])
for c, (la, lo) in CITIES.items():
    if BBOX[0] < lo < BBOX[2] and BBOX[1] < la < BBOX[3]:
        ax.plot(lo, la, "k^", ms=5); ax.text(lo + .3, la + .3, c, fontsize=8)
plt.colorbar(im, label="LST anomalisi (°C)"); ax.set_title("Akdeniz, 12–19 Temmuz 2023 gündüz LST − 2018–2022 ortalaması (MODIS MOD11A2)")
plt.tight_layout(); plt.savefig(OUT / "anomaly_map.png", dpi=90); plt.close()
summary["anomaly_mean_land_c"] = float(np.nanmean(y23 - base))
json.dump(summary, open(OUT / "results.json", "w"), indent=2)
