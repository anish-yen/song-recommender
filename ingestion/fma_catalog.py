"""Download, extract, and index an FMA subset end to end.

    uv run python -m ingestion.fma_catalog --index-dir .index-fma-small --device mps

Steps, each skipped when already done:

1. download ``fma_metadata.zip`` and ``fma_<subset>.zip`` into ``data/audio/fma/``
   (resumable; sha1-verified against the checksums published in the FMA README);
2. extract them with 7-Zip (``brew install sevenzip``) — macOS ``unzip`` cannot read
   their compression method;
3. build the index in **stages** (by default 100, 400, then every track per genre).
   Each stage reuses the previous stage's vectors through the index cache, so an
   interrupted run loses at most one stage, and a usable smaller index exists early.

Measured on an Apple-silicon Mac with ``--device mps``: ~0.25 s per track for CLAP
and ~0.4 s per track for attributes, so fma_small (8,000 tracks) takes about 1.5 h.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://os.unil.cloud.switch.ch/fma/"
#: sha1 sums from https://github.com/mdeff/fma (README, 2017-05-09 release).
SHA1 = {
    "fma_metadata.zip": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
    "fma_small.zip": "ade154f733639d52e35e32f5593efe5be76c6d70",
    "fma_medium.zip": "c67b69ea232021025fca9231fc1c7c1a063ab50b",
}
CHUNK = 1 << 22


def sha1_of(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, dest: Path) -> Path:
    """Fetch ``name`` into ``dest`` (resuming a partial file) and verify its sha1."""
    path = dest / name
    verified = path.with_name(path.name + ".sha1-ok")
    if verified.exists():
        return path
    have = path.stat().st_size if path.exists() else 0
    request = urllib.request.Request(BASE_URL + name, headers={"Range": f"bytes={have}-"} if have else {})
    print(f"downloading {name}" + (f" (resuming at {have / 1e9:.2f} GB)" if have else ""), flush=True)
    try:
        with urllib.request.urlopen(request) as response, open(path, "ab" if have else "wb") as out:
            if have and response.status != 206:  # server ignored the range: start over
                out.truncate(0)
            while chunk := response.read(CHUNK):
                out.write(chunk)
    except urllib.error.HTTPError as exc:
        if exc.code != 416:  # 416: nothing left to fetch
            raise
    actual = sha1_of(path)
    if actual != SHA1[name]:
        raise SystemExit(f"{name}: sha1 {actual} does not match {SHA1[name]}; delete it and retry")
    verified.touch()
    print(f"{name}: sha1 OK", flush=True)
    return path


def extract(archive: Path, expected_dir: Path) -> None:
    if expected_dir.is_dir() and any(expected_dir.iterdir()):
        return
    seven_zip = shutil.which("7zz") or shutil.which("7z")
    if not seven_zip:
        raise SystemExit("7-Zip is required to extract FMA archives: brew install sevenzip")
    print(f"extracting {archive.name}", flush=True)
    subprocess.run([seven_zip, "x", "-y", f"-o{archive.parent}", str(archive)],
                   check=True, stdout=subprocess.DEVNULL)


def run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingestion.fma_catalog",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("--subset", default="small", choices=["small", "medium"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/audio/fma"))
    parser.add_argument("--index-dir", default=".index-fma-small")
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, or auto")
    parser.add_argument("--stages", default="100,400,0",
                        help="tracks per genre for each build stage; 0 = all (default: 100,400,0)")
    args = parser.parse_args(argv)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    metadata_zip = download("fma_metadata.zip", args.data_dir)
    audio_zip = download(f"fma_{args.subset}.zip", args.data_dir)
    extract(metadata_zip, args.data_dir / "fma_metadata")
    extract(audio_zip, args.data_dir / f"fma_{args.subset}")

    manifest = Path("data") / f"manifest.fma-{args.subset}.json"
    for per_genre in (int(s) for s in args.stages.split(",")):
        label = "all tracks" if per_genre == 0 else f"{per_genre} per genre"
        print(f"\n=== stage: {label} ===", flush=True)
        run("-m", "ingestion.fma", "--metadata", str(args.data_dir / "fma_metadata"),
            "--audio", str(args.data_dir / f"fma_{args.subset}"), "--subset", args.subset,
            "--per-genre", str(per_genre), "--out", str(manifest))
        run("-m", "recommendation_engine.demo", "--index-dir", args.index_dir,
            "build-index", "--manifest", str(manifest), "--device", args.device)
        run("-m", "recommendation_engine.attributes", "--index-dir", args.index_dir,
            "--device", args.device)

    print(f"\ndone. Serve it with:\n  INDEX_DIR={args.index_dir} uv run uvicorn api.app:app --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
