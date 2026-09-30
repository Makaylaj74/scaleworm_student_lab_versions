"""Parallel, resumable worm-count harvest over a recording manifest (scene-sort-free).

For every recording: extract frames across the pan window (parallel ffmpeg, CPU), run the
detector on them (batched, GPU), and take the median-of-top-3 richest frames as the count
(validated: r=0.74 vs manual, ~0.72 recall factor). Writes one row per recording, flushing
after every chunk so the job is RESUMABLE — re-running skips recordings already in --out.

Compute notes:
  - EXTRACTION (ffmpeg on ~900 MB files) is the bottleneck; parallelized with a thread pool.
  - WORKERS default 16, HARD-CAPPED at 24 (JupyterHub container limit; >24 has crashed it).
  - Inference is batched on the GPU in the main thread (no multi-process CUDA).

Usage:
  MODEL=99_runs/scaleworm_v5_all/weights/best.pt CONF=0.40 T0=120 T1=600 STEP=60 \
  WORKERS=16 RECALL=0.72 CHUNK=20 \
  /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/harvest_series.py \
    --manifest <in.csv> --out <out.csv>
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from count_frames import count_worms, extract_frame

MODEL = os.environ.get("MODEL", str(REPO / "99_runs/scaleworm_v5_all/weights/best.pt"))
CONF = float(os.environ.get("CONF", "0.40"))
T0 = float(os.environ.get("T0", "120"))
T1 = float(os.environ.get("T1", "600"))
STEP = float(os.environ.get("STEP", "60"))
WORKERS = min(int(os.environ.get("WORKERS", "16")), 24)  # container cap
RECALL = float(os.environ.get("RECALL", "0.72"))  # med-top3 clear recall factor
CHUNK = int(os.environ.get("CHUNK", "20"))  # recordings per checkpoint

FIELDS = [
    "frame_id",
    "datetime_utc",
    "best_time_s",
    "count_raw",
    "count_corrected",
    "max_count",
    "n_frames",
    "all_counts",
]


def load_done(out: Path) -> set[str]:
    if not out.exists():
        return set()
    with out.open() as f:
        return {r["frame_id"] for r in csv.DictReader(f)}


def aggregate(tc: list[tuple[float, int]]) -> dict:
    top = sorted(tc, key=lambda x: -x[1])[:3]
    med = int(statistics.median([c for _, c in top]))
    best_t = min(top, key=lambda x: abs(x[1] - med))[0]
    return {
        "best_time_s": best_t,
        "count_raw": med,
        "count_corrected": round(med / RECALL, 2),
        "max_count": top[0][1],
        "n_frames": len(tc),
        "all_counts": ";".join(str(c) for _, c in tc),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(MODEL)
    names = model.names
    times = [T0 + i * STEP for i in range(int((T1 - T0) / STEP) + 1)]

    done = load_done(args.out)
    all_rows = [r for r in csv.DictReader(args.manifest.open())]
    todo = [
        r
        for r in all_rows
        if r["frame_id"] not in done and Path(r["video_path"]).exists()
    ]
    print(
        f"manifest {len(all_rows)} | already done {len(done)} | to do {len(todo)} "
        f"| window {T0:.0f}-{T1:.0f}/{STEP:.0f}s ({len(times)} frames) | workers {WORKERS}"
    )

    new = not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fout = args.out.open("a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDS)
    if new:
        writer.writeheader()
        fout.flush()

    n_done = 0
    for c0 in range(0, len(todo), CHUNK):
        chunk = todo[c0 : c0 + CHUNK]
        tmp = Path(tempfile.mkdtemp())
        # (recording_index, t, png_path)
        specs = [
            (ri, t, tmp / f"{ri}_{int(t)}.png")
            for ri in range(len(chunk))
            for t in times
        ]

        def _ex(spec, chunk=chunk):
            ri, t, png = spec
            return (
                spec if extract_frame(Path(chunk[ri]["video_path"]), t, png) else None
            )

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            got = [s for s in ex.map(_ex, specs) if s is not None]

        # batched GPU inference over all extracted frames in the chunk
        counts: dict[int, list[tuple[float, int]]] = {}
        if got:
            preds = model([str(p) for _, _, p in got], conf=CONF, verbose=False)
            for (ri, t, _p), res in zip(got, preds):
                c = count_worms([int(x) for x in res.boxes.cls.tolist()], names)
                counts.setdefault(ri, []).append((t, c))

        for ri, r in enumerate(chunk):
            tc = counts.get(ri)
            if not tc:
                continue
            row = {
                "frame_id": r["frame_id"],
                "datetime_utc": r["datetime_utc"],
                **aggregate(tc),
            }
            writer.writerow(row)
            n_done += 1
        fout.flush()
        for p in tmp.glob("*"):
            p.unlink()
        tmp.rmdir()
        print(f"  {c0 + len(chunk)}/{len(todo)} recordings done", flush=True)

    fout.close()
    print(f"harvest complete: {n_done} new recordings -> {args.out}")


if __name__ == "__main__":
    main()
