"""AI worm-count series: run the detector on every sorted Scene-1 frame.

For each frame in frame_manifest.csv this runs the date-appropriate detector at a
PER-REGIME confidence threshold and records the box count as `ai_count` — the
AI-annotation counterpart to the manual box-corrected counts. Uses the already-
extracted PNG in validation/monday_manual_series/images/ when present (fast; no
ffmpeg), else extracts at scene1_time_s.

Default deployment (from the v5 conf sweep, sweep_conf_retrain.py): the single
retrained detector v5_all for BOTH regimes, but at conf 0.40 for clear footage
(date < BLUR_ONSET) and conf 0.25 for blurry footage — blur lowers box confidence,
so the lower cut keeps counts calibrated (clear count-ratio 100%, blurry 95%).

Outputs:
  notebooks/ai_counts_per_frame.csv   (frame_id, date, model, conf, ai_count, in_training)
  notebooks/worm_timeseries_ai_2017_2024.csv  (weekly-Monday mean/SEM)

`in_training` flags frames that were in the detector's train/val split — those must
be EXCLUDED from any manual-vs-AI validation (leakage). The series itself keeps them
(they are still real counts) but the validation step drops them.

Env: V2_MODEL, V3_MODEL (clear/blurry model paths), CONF_CLEAR (0.40),
CONF_BLURRY (0.25) [or CONF for a single threshold], BLUR_ONSET (2023-08-11).
"""

from __future__ import annotations

import argparse
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
# Per-regime conf: fall back to a single CONF if the split values are unset.
_CONF = os.environ.get("CONF", "0.25")
CONF_CLEAR = float(os.environ.get("CONF_CLEAR", _CONF))
CONF_BLURRY = float(os.environ.get("CONF_BLURRY", _CONF))
BLUR_ONSET = pd.Timestamp(os.environ.get("BLUR_ONSET", "2023-08-11")).date()
_V5 = str(REPO / "99_runs/scaleworm_v5_all/weights/best.pt")
MODEL_PATHS = {
    "clear": Path(os.environ.get("V2_MODEL", _V5)),
    "blurry": Path(os.environ.get("V3_MODEL", _V5)),
}


def _training_stems() -> set[str]:
    """Stems in the v5 detector's train/val split (for leakage flagging)."""
    stems = set()
    for ds in ("scaleworm_retrain_all", "scaleworm_retrain_clear"):
        for split in ("train", "val"):
            for p in glob.glob(str(REPO / f"datasets/{ds}/labels/{split}/*.txt")):
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="frame_manifest.csv to score")
    ap.add_argument("--img-dir", type=Path, default=IMG_DIR, help="pre-extracted PNG cache dir")
    ap.add_argument("--out-frame", type=Path, default=OUT_FRAME, help="per-frame ai_count CSV")
    ap.add_argument("--out-series", type=Path, default=OUT_SERIES, help="per-day mean/SEM CSV")
    args = ap.parse_args()

    from ultralytics import YOLO

    models = {k: YOLO(str(v)) for k, v in MODEL_PATHS.items()}
    train = _training_stems()
    df = pd.read_csv(args.manifest)
    rows = []
    for _, r in df.iterrows():
        fid = r["frame_id"]
        d = datetime.fromisoformat(r["datetime_utc"]).date()
        key = "clear" if d < BLUR_ONSET else "blurry"
        conf = CONF_CLEAR if key == "clear" else CONF_BLURRY
        img = args.img_dir / f"{fid}.png"
        tmp = None
        if not img.exists():
            tmp = _extract(r["video_path"], float(r["scene1_time_s"]))
            img = tmp
            if img is None:
                continue
        res = models[key].predict(str(img), conf=conf, verbose=False)
        n = len(res[0].boxes)
        if tmp:
            tmp.unlink(missing_ok=True)
        rows.append(
            {
                "frame_id": fid,
                "date": d.isoformat(),
                "model": key,
                "conf": conf,
                "ai_count": n,
                "in_training": fid in train,
            }
        )
    fr = pd.DataFrame(rows)
    args.out_frame.parent.mkdir(parents=True, exist_ok=True)
    fr.to_csv(args.out_frame, index=False)

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
    pd.DataFrame(out).to_csv(args.out_series, index=False)
    print(
        f"{len(fr)} frames scored (conf clear={CONF_CLEAR}/blurry={CONF_BLURRY}); "
        f"{fr['in_training'].sum()} flagged in-training. "
        f"-> {len(out)} days. wrote {args.out_frame.name}, {args.out_series.name}"
    )


if __name__ == "__main__":
    main()
