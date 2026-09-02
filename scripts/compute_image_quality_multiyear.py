"""Multi-year image-quality survey for CAMHDA301 (Mushroom vent, Axial), 2015-2026.

Extends the 2023-2024 image_quality_timeseries to the full CamHD archive to find
the clearest camera-years/months. For each sampled recording, extract 8 frames at
FIXED time offsets (fast input-seek, low I/O -> only small chunks read, not the
whole ~900 MB file) and compute per-frame Laplacian-variance sharpness plus
brightness/contrast. Fixed offsets sample a consistent set of PTZ-program phases
across clips, so monthly aggregates are comparable across years.

This is a RELATIVE index, not the hand-sorted scene-1 metric: it mixes PTZ scenes,
so absolute values differ from image_quality_timeseries.csv. Validate the monthly
series against the known 2023-sharp / 2024-blurry contrast before trusting earlier
years (see aggregate step).

Resumable: per-recording rows appended to OUT_CSV; reruns skip done paths.
Parallelism: 12 workers (Hub 24-worker cap; single agent).
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

_DEF_MAN = "/tmp/camhd_survey/manifest_multiyear.csv"
_DEF_OUT = (
    "/home/jovyan/scaleworm-student-lab/notebooks/image_quality_multiyear_2015_2026.csv"
)
# optional: python compute_image_quality_multiyear.py <manifest.csv> <out.csv>
MANIFEST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(_DEF_MAN)
OUT_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(_DEF_OUT)
N_FRAMES = 8  # sampled at even fractions 0.1..0.9 of each clip's duration
FRACS = [0.1 + 0.8 * i / (N_FRAMES - 1) for i in range(N_FRAMES)]
BLACK_BRIGHT = 10.0  # frames dimmer than this are dropped (lights off / black feed)
WORKERS = 12
FIELDS = [
    "ym",
    "ts",
    "n_frames",
    "n_black",
    "sharp_median",
    "sharp_p90",
    "sharp_max",
    "brightness",
    "contrast",
    "note",
    "path",
]


def _duration(path: str) -> float | None:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def process(row: dict) -> list:
    path = row["path"]
    dur = _duration(path)
    if dur is None or dur < 5:
        return [
            row["ym"],
            row["ts"],
            0,
            0,
            "",
            "",
            "",
            "",
            "",
            "corrupt/no-duration",
            path,
        ]
    sharp, bright, contr, n_black = [], [], [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, fr in enumerate(FRACS):
            op = f"{tmp}/f{i}.png"
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{dur * fr:.1f}",
                    "-i",
                    path,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    op,
                ],
                check=False,
            )
            img = cv2.imread(op)
            if img is None:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            b = float(g.mean())
            if b < BLACK_BRIGHT:  # black / lights-off frame
                n_black += 1
                continue
            sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
            bright.append(b)
            contr.append(float(g.std()))
    if not sharp:
        note = "all-black" if n_black else "extract-failed"
        return [row["ym"], row["ts"], 0, n_black, "", "", "", "", "", note, path]
    s = np.array(sharp)
    return [
        row["ym"],
        row["ts"],
        len(s),
        n_black,
        f"{np.median(s):.1f}",
        f"{np.percentile(s, 90):.1f}",
        f"{s.max():.1f}",
        f"{np.mean(bright):.2f}",
        f"{np.mean(contr):.2f}",
        "",
        path,
    ]


def main() -> None:
    with MANIFEST.open() as fh:
        todo = list(csv.DictReader(fh))

    done: set[str] = set()
    if OUT_CSV.exists():
        with OUT_CSV.open() as fh:
            done = {r["path"] for r in csv.DictReader(fh)}
    else:
        with OUT_CSV.open("w", newline="") as fh:
            csv.writer(fh).writerow(FIELDS)

    todo = [r for r in todo if r["path"] not in done]
    print(f"to process: {len(todo)}  (already done: {len(done)})", flush=True)

    n = 0
    with OUT_CSV.open("a", newline="") as fh:
        w = csv.writer(fh)
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(process, r): r for r in todo}
            for fut in as_completed(futs):
                w.writerow(fut.result())
                n += 1
                if n % 25 == 0:
                    fh.flush()
                    print(f"  {n}/{len(todo)}", flush=True)
    print(f"DONE {n} new rows -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
