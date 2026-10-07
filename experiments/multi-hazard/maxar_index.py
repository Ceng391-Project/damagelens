import json
import posixpath
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://maxar-opendata.s3.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23/"
OUT = Path(__file__).parent.parent.parent / "data" / "maxar_tr" / "index.json"


def load(u):
    return json.load(urllib.request.urlopen(u))


def join(base_url, href):
    return posixpath.normpath(posixpath.join(posixpath.dirname(base_url), href)).replace("https:/", "https://")


def items(u):
    return [join(u, l["href"]) for l in load(u)["links"] if l["rel"] == "item"]


def meta(u):
    j = load(u); p = j["properties"]; v = j["assets"].get("visual", {}).get("href")
    return dict(url=u, id=j["id"], dt=p["datetime"], bbox=j["bbox"], qk=p.get("quadkey"), gsd=p.get("gsd"),
                off=p.get("view:off_nadir"), cloud=p.get("tile:clouds_percent"),
                visual=(v if not v or v.startswith("http") else join(u, v)))


if __name__ == "__main__":
    cols = [BASE + l["href"].lstrip("./") for l in load(BASE + "collection.json")["links"] if l["rel"] == "child"]
    with ThreadPoolExecutor(16) as ex:
        urls = sum(ex.map(items, cols), [])
    with ThreadPoolExecutor(24) as ex:
        rows = list(ex.map(meta, urls))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT, "w"))
    print(len(rows), "tiles ->", OUT)
