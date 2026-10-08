import argparse
import json
from collections import Counter

from .. import REPO_ROOT
from .dataset import read_annotation, to_geojson, validate_area
from .prepare import ensure_tiles, prepare_area

ROOT = REPO_ROOT / "labels"


def areas(name=None):
    return sorted(d for d in ROOT.iterdir() if (d / "manifest.json").exists() and (not name or d.name == name)) if ROOT.exists() else []


def main(argv=None):
    ap = argparse.ArgumentParser(prog="damagelens-label", description="Hand-label building damage on raw Maxar pre/post tiles")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="cut a new area into aligned 512 px pre/post tiles")
    p.add_argument("name"); p.add_argument("--lon", type=float, required=True); p.add_argument("--lat", type=float, required=True)
    p.add_argument("--side-m", type=float, default=1536); p.add_argument("--gsd", type=float, default=0.5)
    p.add_argument("--pre-date"); p.add_argument("--post-date")
    s = sub.add_parser("serve", help="open the labeling UI for an area (regenerates missing tiles from its manifest)")
    s.add_argument("name"); s.add_argument("--port", type=int, default=8765)
    e = sub.add_parser("export", help="write <area>/labels.geojson (lon/lat polygons)")
    e.add_argument("name")
    st = sub.add_parser("status", help="progress per area")
    st.add_argument("name", nargs="?")
    st.add_argument("--markdown", action="store_true", help="print a Markdown table (for CI summaries)")
    v = sub.add_parser("validate", help="check annotation files against the manifest; exits 1 on errors")
    v.add_argument("name", nargs="?")
    a = ap.parse_args(argv)

    if a.cmd == "prepare":
        out, m = prepare_area(a.name, a.lon, a.lat, a.side_m, a.gsd, a.pre_date, a.post_date, ROOT)
        print(f"{len(m['tiles'])} tiles, pre {m['pre_date']}, post {m['post_date']} -> {out}")
    elif a.cmd == "serve":
        from .server import serve
        ensure_tiles(ROOT / a.name); serve(ROOT / a.name, a.port)
    elif a.cmd == "export":
        gj = to_geojson(a.name, ROOT); f = ROOT / a.name / "labels.geojson"
        json.dump(gj, open(f, "w")); print(f"{len(gj['features'])} polygons -> {f}")
    elif a.cmd == "status":
        if a.markdown:
            print("| area | tiles | done | skipped | todo | damaged | destroyed | intact |\n|---|---|---|---|---|---|---|---|")
        for area in areas(a.name):
            m = json.load(open(area / "manifest.json"))
            anns = [read_annotation(area, t["id"]) for t in m["tiles"]]
            st = Counter(x.get("status", "todo") for x in anns)
            cls = Counter(p["cls"] for x in anns if x.get("status") == "done" for p in x["polygons"])
            if a.markdown:
                print(f"| {area.name} | {len(anns)} | {st['done']} | {st['skip']} | {st['todo']} | {cls['damaged']} | {cls['destroyed']} | {cls['intact']} |")
            else:
                print(f"{area.name}: {len(anns)} tiles | done {st['done']} skip {st['skip']} todo {st['todo']} | polygons {dict(cls)}")
    elif a.cmd == "validate":
        errors = [e for area in areas(a.name) for e in validate_area(area)]
        print("\n".join(errors) if errors else f"labels ok ({len(areas(a.name))} areas)")
        raise SystemExit(1 if errors else 0)

if __name__ == "__main__":
    main()
