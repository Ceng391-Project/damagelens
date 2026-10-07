import json
import re
from datetime import datetime, UTC
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .dataset import CLASS_VALUE, read_annotation

STATIC = Path(__file__).parent / "static"
TILE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, area: Path, **kw):
        self.area = area
        super().__init__(*a, directory=str(STATIC), **kw)

    def log_message(self, *a):
        pass

    def end_headers(self):
        if not self.path.startswith("/tiles/"):
            self.send_header("Cache-Control", "no-cache")  # the UI changes between versions; tiles never do
        super().end_headers()

    def _json(self, obj, code=HTTPStatus.OK):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _tile_id(self, s):
        if not TILE_ID.match(s) or not (self.area / "tiles" / f"{s}_post.png").exists():
            self._json(dict(error="unknown tile"), HTTPStatus.NOT_FOUND); return None
        return s

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/manifest":
            m = json.load(open(self.area / "manifest.json"))
            m["status"] = {t["id"]: read_annotation(self.area, t["id"]).get("status", "todo") for t in m["tiles"]}
            return self._json(m)
        if p.startswith("/api/ann/"):
            tid = self._tile_id(p.rsplit("/", 1)[1])
            return tid and self._json(read_annotation(self.area, tid))
        if p.startswith("/tiles/"):
            name = p.rsplit("/", 1)[1]
            f = self.area / "tiles" / name
            if not re.match(r"^[A-Za-z0-9_\-]+_(pre|post)\.png$", name) or not f.exists():
                return self._json(dict(error="not found"), HTTPStatus.NOT_FOUND)
            data = f.read_bytes()
            self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "max-age=3600"); self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data); return
        return super().do_GET()

    def do_POST(self):
        p = self.path.split("?")[0]
        if not p.startswith("/api/ann/"):
            return self._json(dict(error="not found"), HTTPStatus.NOT_FOUND)
        tid = self._tile_id(p.rsplit("/", 1)[1])
        if not tid:
            return
        try:
            ann = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            polys = [dict(cls=q["cls"], points=[[round(float(x), 1), round(float(y), 1)] for x, y in q["points"]])
                     for q in ann.get("polygons", []) if q.get("cls") in CLASS_VALUE and len(q.get("points", [])) >= 3]
            status = ann.get("status", "todo")
            assert status in ("todo", "done", "skip")
        except (ValueError, KeyError, TypeError, AssertionError):
            return self._json(dict(error="bad annotation"), HTTPStatus.BAD_REQUEST)
        out = dict(tile=tid, status=status, user=str(ann.get("user", ""))[:40], polygons=polys,
                   updated=datetime.now(UTC).isoformat(timespec="seconds"))
        f = self.area / "annotations" / f"{tid}.json"; f.parent.mkdir(exist_ok=True)
        tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(out, indent=1)); tmp.replace(f)
        return self._json(dict(ok=True, updated=out["updated"]))


def serve(area: Path, port: int = 8765):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, area=area))
    print(f"DamageLens labeler: http://127.0.0.1:{port}/  (area {area.name}, Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
