import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from damagelens import REPO_ROOT, RUNS_DIR
from damagelens.data.maxar import items_near, load_index, mosaic, pick_dates, snow_cloud_fraction
from damagelens.models import load_model
from damagelens.summarize import damage_map, grid_summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="End-to-end demo on Maxar Open Data: mosaic → align → model → damage grid")
    ap.add_argument("--lon", type=float, default=36.925)
    ap.add_argument("--lat", type=float, default=37.585)
    ap.add_argument("--side-m", type=float, default=1536)
    ap.add_argument("--gsd", type=float, default=0.5)
    ap.add_argument("--cell-m", type=float, default=48)
    ap.add_argument("--model", type=Path, default=REPO_ROOT / "experiments/feasibility/outputs/kate_only.pt")
    ap.add_argument("--name", default="kahramanmaras")
    a = ap.parse_args(argv)

    pre_d, post_d = items_near(load_index(), a.lon, a.lat)
    d_pre, d_post = pick_dates(pre_d, post_d, a.lon, a.lat, a.side_m)
    pre = mosaic(pre_d[d_pre], a.lon, a.lat, a.side_m, a.gsd)
    post = mosaic(post_d[d_post], a.lon, a.lat, a.side_m, a.gsd)
    model, meta = load_model(a.model); thr = meta.get("thr", 0.5)
    prob, shifts = damage_map(model, pre, post)
    mask = prob > thr
    grid = grid_summary(mask, int(round(a.cell_m / a.gsd)))

    out = RUNS_DIR / "demo" / a.name; out.mkdir(parents=True, exist_ok=True)
    res = dict(lon=a.lon, lat=a.lat, side_m=a.side_m, gsd=a.gsd, pre_date=d_pre, post_date=d_post, thr=thr,
               post_snow_cloud=snow_cloud_fraction(post), damaged_frac=float(mask.mean()),
               cells_over_5pct=float((grid > 0.05).mean()), median_tile_shift_px=float(np.nanmedian(shifts)),
               max_tile_shift_px=float(np.nanmax(shifts)))
    json.dump(res, open(out / "summary.json", "w"), indent=2)

    ext = [0, a.side_m, 0, a.side_m]
    fig, ax = plt.subplots(1, 4, figsize=(24, 6.4))
    ax[0].imshow(pre, extent=ext); ax[0].set_title(f"pre {d_pre}")
    ax[1].imshow(post, extent=ext); ax[1].set_title(f"post {d_post}")
    ov = post.copy(); ov[mask] = (0.45 * ov[mask] + [140, 0, 0]).astype(np.uint8)
    ax[2].imshow(ov, extent=ext); ax[2].set_title(f"damage prediction (thr {thr:.2f})")
    im = ax[3].imshow(grid, cmap="inferno", vmin=0, vmax=max(.3, float(grid.max())), extent=ext)
    ax[3].set_title(f"damaged area fraction per {a.cell_m:.0f} m cell"); plt.colorbar(im, ax=ax[3], fraction=.046)
    for q in ax: q.set_xlabel("m"); q.set_ylabel("m")
    plt.tight_layout(); plt.savefig(out / "map.png", dpi=70); plt.close()
    print(json.dumps(res, indent=1)); print("->", out)


if __name__ == "__main__":
    main()
