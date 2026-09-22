"""Materialize the Scene-1 classifier frames listed in labels.csv.

Reads `datasets/scene_classifier/labels.csv` (from build_scene_classifier_dataset.py)
and extracts one JPG per row at its `tile_time_s` into

    datasets/scene_classifier/images/<split>/<label>/<stem>_t<time>.jpg

so an ImageFolder-style loader (or a frozen-backbone embedding pass) can consume it
directly. Frames are downscaled to --scale-width (default 640 px; embedding backbones
resize to ~224 anyway) to keep the set small. Resumable: existing JPGs are skipped.
Workers are clamped to the host ceiling (24) per the lab cap.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "datasets/scene_classifier/labels.csv"
IMG_ROOT = REPO / "datasets/scene_classifier/images"
MAX_WORKERS = 24


def out_path(row: dict, root: Path) -> Path:
    return (
        root / row["split"] / row["label"] / f"{row['stem']}_t{row['tile_time_s']}.jpg"
    )


def extract_one(row: dict, root: Path, scale_width: int) -> tuple[str, bool]:
    dst = out_path(row, root)
    if dst.exists() and dst.stat().st_size > 0:
        return (str(dst), True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(row["tile_time_s"]),
        "-i",
        row["video_path"],
        "-frames:v",
        "1",
        "-vf",
        f"scale={scale_width}:-2",
        "-q:v",
        "3",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, check=False)
    ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    return (str(dst), ok)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--img-root", type=Path, default=IMG_ROOT)
    ap.add_argument("--scale-width", type=int, default=640)
    ap.add_argument("--workers", type=int, default=12, help=f"Max {MAX_WORKERS}.")
    ap.add_argument("--limit", type=int, default=None, help="Cap rows (smoke test).")
    args = ap.parse_args()

    workers = min(args.workers, MAX_WORKERS)
    with args.labels.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    ok = miss = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(extract_one, r, args.img_root, args.scale_width) for r in rows
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            _, good = fut.result()
            ok += good
            miss += not good
            if i % 500 == 0:
                print(f"[{i}/{len(rows)}] ok={ok} missing={miss}", flush=True)
    print(
        f"done: {ok} extracted/present, {miss} missing (of {len(rows)}) -> {args.img_root}"
    )


if __name__ == "__main__":
    main()
