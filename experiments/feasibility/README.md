# DamageLens · feasibility — #30 on the 6 February 2023 earthquakes (KATE-CD)

- `common.py` — loaders for KATE-CD / xBD parquet, pixel metrics
- `01_stats_baseline.py` — KATE-CD stats + image-difference baseline (threshold picked on val)
- `02_train.py` — 6-channel U-Net (ResNet18) on pre+post; `--train kate|xbd`, `--init` for fine-tune
- `run_all.sh` — the three training runs (KATE only, xBD only, xBD → KATE)
- `03_figure.py` — prediction figure for the three models
- `04_gaps.py` — classical baselines (CVA, 1−SSIM, PCA-kmeans, Otsu) + alignment sensitivity and phase-correlation registration
- `05_spatial_summary.py` — end-to-end on raw Maxar Open Data (Kahramanmaraş centre): mosaic → registration → model → 48 m damage grid; needs `../../download_data.sh maxar`
- `data/`, `outputs/` — gitignored; fill with `../../download_data.sh kate` (data now lives in the repo-root `data/`)
