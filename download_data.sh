#!/usr/bin/env bash
# Usage: ./download_data.sh [kate|xbd|flood|landslide|valencia|maxar|all]   (default: kate)
set -euo pipefail
cd "$(dirname "$0")"
HF=https://huggingface.co/datasets
what=${1:-kate}

get() { [ -s "$2" ] && return 0; curl -sfL --retry 5 -o "$2.part" "$1" && mv "$2.part" "$2"; }

if [[ $what == kate || $what == all ]]; then  # ~450 MB
  mkdir -p feasibility/data/kate-cd
  for s in train validation test; do get "$HF/CSCRS/kate-cd/resolve/main/data/$s-00000-of-00001.parquet" feasibility/data/kate-cd/$s.parquet; done
fi
if [[ $what == xbd || $what == all ]]; then  # ~24 GB raw; disaster-eval/xbd_prep.py turns it into ~9 GB of 512 px memmaps
  mkdir -p disaster-eval/data/xbd
  curl -s "https://huggingface.co/api/datasets/hannan022/xview2-xbd/tree/main/data" | python -c "import json,sys;[print(x['path'].split('/')[-1]) for x in json.load(sys.stdin)]" |
    xargs -P 4 -I{} sh -c '[ -s disaster-eval/data/xbd/{} ] || curl -sfL --retry 5 -o disaster-eval/data/xbd/{} '"$HF"'/hannan022/xview2-xbd/resolve/main/data/{}'
fi
if [[ $what == flood || $what == all ]]; then  # ~1.8 GB
  mkdir -p disaster-eval/data/sen1floods11
  get "$HF/blumenstiel/Sen1Floods11/resolve/main/sen1floods11_v1_1.tar.gz" disaster-eval/data/sen1floods11/s1f11.tar.gz
  tar xzf disaster-eval/data/sen1floods11/s1f11.tar.gz -C disaster-eval/data/sen1floods11 && rm disaster-eval/data/sen1floods11/s1f11.tar.gz
fi
if [[ $what == landslide || $what == all ]]; then  # Landslide4Sense ~0.5 GB + UAV subset ~2 GB
  python -c "from huggingface_hub import snapshot_download as s; s('ibm-nasa-geospatial/Landslide4sense', repo_type='dataset', local_dir='disaster-eval/data/landslide4sense', max_workers=8)"
  mkdir -p disaster-eval/data/landslide_uav
  for f in test-00000-of-00006 test-00001-of-00006 validation-00000-of-00005 train-00000-of-00019 train-00001-of-00019 train-00002-of-00019 train-00003-of-00019; do
    get "$HF/syeddhasnainn/landslide-uav-all/resolve/main/data/$f.parquet" disaster-eval/data/landslide_uav/$f.parquet; done
fi
if [[ $what == valencia || $what == all ]]; then  # Copernicus EMS EMSR773 delineation, ~23 MB
  mkdir -p disaster-eval/data/valencia
  get "https://rapidmapping.emergency.copernicus.eu/backend/EMSR773/AOI01/DEL_PRODUCT/EMSR773_AOI01_DEL_PRODUCT_v1.zip" disaster-eval/data/valencia/del.zip
  (cd disaster-eval/data/valencia && unzip -o -q del.zip)
fi
if [[ $what == maxar || $what == all ]]; then  # STAC index of Maxar Open Data, Kahramanmaraş 2023 (imagery is read on demand)
  python disaster-eval/maxar_index.py
fi
# hail (MRMS MESH), heat (MODIS + Meteostat), fire (Sentinel-2 + perimeters) download on demand inside their scripts.
