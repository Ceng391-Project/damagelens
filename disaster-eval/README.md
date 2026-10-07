# DamageLens · disaster-eval — per-disaster satellite/aerial experiments

Run order (all write to `outputs/`, then `build_reports.py` + `build_html.py` write `../docs/disaster-report/`):
- `xbd_prep.py` → `xbd_train.py` → `xbd_eval.py` — 5-class building damage U-Net on xBD (all disaster types) + KATE-CD zero-shot
- `flood_sen1floods11.py`, `flood_valencia.py` — water mapping, Valencia 2024 vs Copernicus EMS EMSR773
- `landslide_l4s.py`, `landslide_uav.py` — Landslide4Sense (satellite) and UAV landslide set
- `tornado_rollingfork.py` — xBD model on NAIP aerial vs NWS DAT EF points
- `hail_nebraska.py <config>` — Sentinel-2 ΔNDVI vs MRMS MESH (configs A/B/C)
- `heat_lst.py` — MODIS LST vs Meteostat Tmax, 10 cities, summer 2023
- `fire_dnbr.py` — Sentinel-2 dNBR vs NIFC/EFFIS perimeters
- `earthquake_maxar_random.py` — random Maxar Open Data tiles, 6 Şubat
- shared: `seg.py` (binary U-Net train/predict), `eo.py` (Planetary Computer reads)
- `maxar_index.py` — builds `data/maxar_tr/index.json` (STAC index of Maxar Open Data, Kahramanmaraş 2023)
- `gpu_queue.sh` — runs the GPU jobs one after another (two at once on a 16 GB Mac swaps heavily)
- `data/`, `outputs/` — gitignored; fill with `../download_data.sh <set>`
