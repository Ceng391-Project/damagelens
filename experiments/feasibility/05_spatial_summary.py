import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy.ndimage import shift as nd_shift
from skimage.color import rgb2gray
from skimage.registration import phase_cross_correlation

from common import OUT

ROOT = Path(__file__).parent
meta = json.load(open(ROOT.parent.parent / "data/maxar_tr/index.json"))
CENTER = (36.925, 37.585)  # Kahramanmaraş city centre (lon, lat)
SIDE_M, CELL_M, T = 1536, 48, 512
UTM = "EPSG:32637"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
cx, cy = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform(*CENTER)
lon0, lat0, lon1, lat1 = CENTER[0] - .012, CENTER[1] - .01, CENTER[0] + .012, CENTER[1] + .01
cands = [m for m in meta if m["visual"] and m["bbox"][0] < lon1 and m["bbox"][2] > lon0 and m["bbox"][1] < lat1 and m["bbox"][3] > lat0]
pre_c = sorted([m for m in cands if m["dt"] < "2023-02-06"], key=lambda m: m["dt"])
post_c = sorted([m for m in cands if m["dt"] >= "2023-02-06"], key=lambda m: m["dt"])


def mosaic(items, gsd):
    n = int(SIDE_M / gsd); tr = from_origin(cx - SIDE_M / 2, cy + SIDE_M / 2, gsd, gsd)
    out = np.zeros((n, n, 3), np.uint8)
    for m in items:
        with rasterio.open(m["visual"]) as src, WarpedVRT(src, crs=UTM, transform=tr, width=n, height=n, resampling=Resampling.average) as v:
            a = v.read([1, 2, 3]).transpose(1, 2, 0)
        fill = (out.sum(-1) == 0) & (a.sum(-1) > 0); out[fill] = a[fill]
    return out


def bright(x):
    return float(((x.min(-1) > 190) & (x.max(-1) - x.min(-1) < 30)).mean())


by_date = lambda c: {d: [m for m in c if m["dt"][:10] == d] for d in sorted({m["dt"][:10] for m in c})}
pre_d, post_d = by_date(pre_c), by_date(post_c)
choice = {}
for name, dd in [("pre", pre_d), ("post", post_d)]:
    best = None
    for d, items in dd.items():
        x = mosaic(items, 2.0); cov = float((x.sum(-1) > 0).mean()); b = bright(x)
        print(name, d, "coverage", round(cov, 2), "snow/cloud", round(b, 3), flush=True)
        score = cov - 3 * b - (0.0 if name == "pre" else 0.002 * (int(d[5:7]) * 31 + int(d[8:])))
        if cov > .9 and (best is None or score > best[0]):
            best = (score, d)
    choice[name] = best[1]
pre_items = pre_d[choice["pre"]]; post_items = post_d[choice["post"]]
print("chosen", choice, flush=True)

kate = smp.Unet("resnet18", encoder_weights=None, in_channels=6, classes=1).to(dev)
kate.load_state_dict(torch.load(OUT / "kate_only.pt", map_location=dev)); kate.eval()
kthr = json.load(open(OUT / "kate_only.json"))["thr"]
mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=dev).view(1, 6, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225] * 2, device=dev).view(1, 6, 1, 1)
results, keep = {}, {}
for gsd in (0.5, 0.35):
    A, B = mosaic(pre_items, gsd), mosaic(post_items, gsd)
    n = A.shape[0]; prob = np.zeros((n, n), np.float32); shifts = []
    Breg = B.copy()
    for i in range(0, n - T + 1, T):
        for j in range(0, n - T + 1, T):
            a, b = A[i:i + T, j:j + T], B[i:i + T, j:j + T]
            sh = phase_cross_correlation(rgb2gray(a), rgb2gray(b), upsample_factor=4)[0]
            shifts.append(float(np.hypot(*sh)))
            if np.hypot(*sh) < 40:
                b = np.clip(nd_shift(b.astype(np.float32), (sh[0], sh[1], 0), order=1, mode="nearest"), 0, 255).astype(np.uint8)
            Breg[i:i + T, j:j + T] = b
            x = torch.from_numpy(np.concatenate([a, b], -1)[None]).to(dev).permute(0, 3, 1, 2).float()
            with torch.no_grad():
                prob[i:i + T, j:j + T] = torch.sigmoid(kate((x / 255 - mean) / std))[0, 0].cpu().numpy()
    c = int(CELL_M / gsd); k = n // c
    grid = lambda m, k=k, c=c: m[: k * c, : k * c].reshape(k, c, k, c).mean((1, 3))
    r = dict(gsd=gsd, tile_shift_px_median=float(np.median(shifts)), tile_shift_px_max=float(np.max(shifts)),
             damaged_frac_at_val_thr=float((prob > kthr).mean()), damaged_frac_at_0_5=float((prob > .5).mean()),
             damaged_frac_at_0_3=float((prob > .3).mean()), prob_p99=float(np.percentile(prob, 99)),
             cells_with_damage_val_thr=float((grid(prob > kthr) > .05).mean()))
    results[f"gsd_{gsd}"] = r; keep[gsd] = (A, Breg, prob, grid)
    print(r, flush=True)
res = dict(center=CENTER, side_m=SIDE_M, cell_m=CELL_M, pre_date=choice["pre"], post_date=choice["post"],
           pre_snow_cloud=bright(keep[0.5][0]), post_snow_cloud=bright(keep[0.5][1]), kate_val_thr=kthr, runs=results)
json.dump(res, open(OUT / "05_spatial_summary.json", "w"), indent=2)

gsd = 0.35 if results["gsd_0.35"]["damaged_frac_at_0_5"] > results["gsd_0.5"]["damaged_frac_at_0_5"] else 0.5
A, B, prob, grid = keep[gsd]
thr_show = kthr if (prob > kthr).mean() > 1e-3 else 0.5
G = grid(prob > thr_show)
ext = [0, SIDE_M, 0, SIDE_M]
fig, ax = plt.subplots(1, 4, figsize=(24, 6.4))
ax[0].imshow(A, extent=ext); ax[0].set_title(f"Maxar öncesi {choice['pre']}")
ax[1].imshow(B, extent=ext); ax[1].set_title(f"Maxar sonrası {choice['post']} (faz korelasyonu ile hizalı)")
ov = B.copy(); m = prob > thr_show; ov[m] = (0.45 * ov[m] + [140, 0, 0]).astype(np.uint8)
ax[2].imshow(ov, extent=ext); ax[2].set_title(f"KATE-CD modeli piksel tahmini (eşik {thr_show:.2f}, {gsd} m/px)")
im = ax[3].imshow(G, cmap="inferno", vmin=0, vmax=max(.3, float(G.max())), extent=ext)
ax[3].set_title(f"{CELL_M} m hücre başına hasarlı alan oranı"); plt.colorbar(im, ax=ax[3], fraction=.046)
for a in ax: a.set_xlabel("m"); a.set_ylabel("m")
plt.suptitle(f"Kahramanmaraş merkezi {SIDE_M}×{SIDE_M} m: ham görüntüden mekânsal hasar özeti (uçtan uca hat)")
plt.tight_layout(); plt.savefig(OUT / "05_spatial_summary.png", dpi=70); plt.close()
print(json.dumps(res, indent=1))
