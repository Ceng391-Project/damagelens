cd "$(dirname "$0")"
while pgrep -f xbd_train.py >/dev/null; do sleep 20; done
export PYTHONUNBUFFERED=1
uv run python xbd_eval.py > outputs/xbd_eval.log 2>&1; echo "xbd_eval $?"
uv run python tornado_rollingfork.py > outputs/tornado.log 2>&1; echo "tornado $?"
uv run python earthquake_maxar_random.py > outputs/earthquake_maxar.log 2>&1; echo "maxar $?"
uv run python landslide_l4s.py > outputs/landslide.log 2>&1; echo "l4s $?"
uv run python landslide_uav.py > outputs/landslide_uav.log 2>&1; echo "uav $?"
