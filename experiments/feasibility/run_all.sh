set -e
cd "$(dirname "$0")"
python 02_train.py --train kate --epochs 30 --name kate_only
python 02_train.py --train xbd --epochs 10 --name xbd_only
python 02_train.py --train kate --epochs 30 --init xbd_only --name xbd_then_kate
