"""AI worm-count series: run the detector on every sorted Scene-1 frame.

For each frame in frame_manifest.csv this runs the date-appropriate detector
(v2 clear-specialist for date < BLUR_ONSET, v3 blurry-generalist otherwise) at
CONF and records the box count as `ai_count` — the AI-annotation counterpart to
the manual box-corrected counts. Uses the already-extracted PNG in
validation/monday_manual_series/images/ when present (fast; no ffmpeg), else
extracts at scene1_time_s.

Outputs:
  notebooks/ai_counts_per_frame.csv   (frame_id, date, model, ai_count, in_training)
  notebooks/worm_timeseries_ai_2017_2024.csv  (weekly-Monday mean/SEM)

`in_training` flags frames that were in the v2/v3 training or val split — those
must be EXCLUDED from any manual-vs-AI validation (leakage). The series itself
keeps them (they are still real counts) but the validation step drops them.

Env: V2_MODEL, V3_MODEL, CONF (0.25), BLUR_ONSET (2023-08-10).
"""

from __future__ import annotations

import glob
import math
import os
import statistics
import subprocess
import tempfile
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "validation/monday_manual_series/frame_manifest.csv"
IMG_DIR = REPO / "validation/monday_manual_series/images"
OUT_FRAME = REPO / "notebooks/ai_counts_per_frame.csv"
OUT_SERIES = REPO / "notebooks/worm_timeseries_ai_2017_2024.csv"
CONF = float(os.environ.get("CONF", "0.25"))
BLUR_ONSET = pd.Timestamp(os.environ.get("BLUR_ONSET", "2023-08-10")).date()
MODEL_PATHS = {
    "v2": Path(
        os.environ.get("V2_MODEL", str(REPO / "99_runs/scaleworm_v2/weights/best.pt"))
    ),
    "v3": Path(
        os.environ.get("V3_MODEL", str(REPO / "99_runs/scaleworm_v3/weights/best.pt"))
    ),
}


def _training_stems() -> set[str]:
    stems = set()
    for split in ("train", "val"):
        for p in glob.glob(str(REPO / f"datasets/scaleworm_v2/labels/{split}/*.txt")):
            stems.add(Path(p).stem)
    return stems


def _extract(video: str, t: float) -> Path | None:
    tmp = Path(tempfile.mkstemp(suffix=".png")[1])
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(t),
        "-i",
        video,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(tmp),
    ]
    r = subprocess.run(cmd, capture_output=True, check=False)
    return tmp if r.returncode == 0 and tmp.stat().st_size > 0 else None


def main() -> None:
    from ultralytics import YOLO

    models = {k: YOLO(str(v)) for k, v in MODEL_PATHS.items()}
    train = _training_stems()
    df = pd.read_csv(MANIFEST)
    rows = []
    for _, r in df.iterrows():
        fid = r["frame_id"]
        d = datetime.fromisoformat(r["datetime_utc"]).date()
        key = "v2" if d < BLUR_ONSET else "v3"
        img = IMG_DIR / f"{fid}.png"
        tmp = None
        if not img.exists():
            tmp = _extract(r["video_path"], float(r["scene1_time_s"]))
            img = tmp
            if img is None:
                continue
        res = models[key].predict(str(img), conf=CONF, verbose=False)
        n = len(res[0].boxes)
        if tmp:
            tmp.unlink(missing_ok=True)
        rows.append(
            {
                "frame_id": fid,
                "date": d.isoformat(),
                "model": key,
                "ai_count": n,
                "in_training": fid in train,
            }
        )
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT_FRAME, index=False)

    # weekly-Monday aggregation of ai_count
    by_day: dict[date, list[int]] = defaultdict(list)
    for _, r in fr.iterrows():
        by_day[date.fromisoformat(r["date"])].append(int(r["ai_count"]))
    out = []
    for d in sorted(by_day):
        c = by_day[d]
        n = len(c)
        mean = statistics.fmean(c)
        sem = statistics.stdev(c) / math.sqrt(n) if n > 1 else float("nan")
        out.append(
            {
                "date": d.isoformat(),
                "n_slots": n,
                "mean_ai": round(mean, 4),
                "sem_ai": round(sem, 4) if not math.isnan(sem) else "",
            }
        )
    pd.DataFrame(out).to_csv(OUT_SERIES, index=False)
    print(
        f"{len(fr)} frames scored (conf={CONF}); "
        f"{fr['in_training'].sum()} flagged in-training. "
        f"-> {len(out)} Mondays. wrote {OUT_FRAME.name}, {OUT_SERIES.name}"
    )


if __name__ == "__main__":
    main()
