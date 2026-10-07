import json
import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image

from damagelens.label.dataset import load_labeled, rasterize, to_geojson
from damagelens.label.server import Handler


def make_area(tmp_path, name="toy", size=64):
    area = tmp_path / name; (area / "tiles").mkdir(parents=True); (area / "annotations").mkdir()
    for k in range(2):
        tid = f"{name}_{k:03d}"
        for kind in ("pre", "post"):
            Image.fromarray(np.full((size, size, 3), 40 * (k + 1), np.uint8)).save(area / "tiles" / f"{tid}_{kind}.png")
    m = dict(name=name, lon=36.9, lat=37.6, side_m=64, gsd=0.5, crs="EPSG:32637", tile=size, pre_date="2022-07-26",
             post_date="2023-02-08", tiles=[dict(id=f"{name}_{k:03d}", row=0, col=k * size, x0=300000.0 + k * 32, y0=4160000.0) for k in range(2)])
    json.dump(m, open(area / "manifest.json", "w"))
    return area


def test_rasterize_classes():
    polys = [dict(cls="intact", points=[[0, 0], [30, 0], [30, 30], [0, 30]]),
             dict(cls="destroyed", points=[[10, 10], [20, 10], [20, 20], [10, 20]])]
    m = rasterize(polys, 32)
    assert m[5, 5] == 3 and m[15, 15] == 2 and m[31, 31] == 0


def test_server_roundtrip_and_dataset(tmp_path):
    area = make_area(tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, area=area))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        man = json.load(urllib.request.urlopen(f"{base}/api/manifest"))
        assert man["status"] == {"toy_000": "todo", "toy_001": "todo"}
        body = json.dumps(dict(status="done", user="ali", polygons=[dict(cls="damaged", points=[[8, 8], [24, 8], [24, 24], [8, 24]]),
                                                                    dict(cls="bogus", points=[[0, 0], [1, 1], [2, 2]])])).encode()
        req = urllib.request.Request(f"{base}/api/ann/toy_000", data=body, headers={"Content-Type": "application/json"}, method="POST")
        assert json.load(urllib.request.urlopen(req))["ok"]
        bad = urllib.request.Request(f"{base}/api/ann/..%2Fetc", data=body, method="POST")
        try:
            urllib.request.urlopen(bad); assert False
        except urllib.error.HTTPError as e:
            assert e.code == 404
        saved = json.load(urllib.request.urlopen(f"{base}/api/ann/toy_000"))
        assert saved["status"] == "done" and saved["user"] == "ali" and len(saved["polygons"]) == 1
        assert urllib.request.urlopen(f"{base}/tiles/toy_000_post.png").headers["Content-Type"] == "image/png"
    finally:
        httpd.shutdown()
    pairs = load_labeled("toy", root=tmp_path)
    assert len(pairs) == 1 and pairs[0].mask[16, 16] == 1 and pairs[0].mask[0, 0] == 0
    gj = to_geojson("toy", root=tmp_path)
    lon, lat = gj["features"][0]["geometry"]["coordinates"][0][0]
    assert 30 < lon < 40 and 30 < lat < 45
