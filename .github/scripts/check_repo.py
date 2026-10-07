import fnmatch
import subprocess
import sys
from pathlib import Path

MAX_MB = 10
FORBIDDEN = [
    "data/*", "runs/*", "*/outputs/*", "outputs/*", "labels/*/tiles/*", "labels/*/labels.geojson",
    "*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.npy", "*.npz", "*.parquet", "*.h5", "*.tif", "*.tiff", "*.grib2",
    "CLAUDE.md", "AGENTS.md", "notes.md", "const.md", "backlog/*", "reports/*", "assets/*", "graphify-out/*",
    "requirements.txt", ".env", "*.env",
]

files = subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True, check=True).stdout.split("\0")
problems = []
for f in filter(None, files):
    if any(fnmatch.fnmatch(f, p) for p in FORBIDDEN):
        problems.append(f"{f}: must not be committed (data, weights, tiles or local working file)")
    elif Path(f).is_file() and Path(f).stat().st_size > MAX_MB * 2**20:
        problems.append(f"{f}: larger than {MAX_MB} MB")
print("\n".join(problems) if problems else f"repo hygiene ok ({len(files)} tracked files)")
sys.exit(1 if problems else 0)
