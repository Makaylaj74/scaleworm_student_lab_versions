"""Gate-driven, recall-corrected AI worm counts for old camera units.

Unlike ``run_ai_counts.py`` (which routes the detector purely by DATE and emits
RAW box counts), this routes by the *per-unit gate result* and applies that
unit's empirical recall correction. For each camera unit that has a
``validation/unit_gates/<unit>/gate_result.csv``:

  * pick the recommended model exactly as ``score_unit_gate.py`` does
    (highest-recall model whose recall >= MIN_RECALL, tie-broken by best
    count-MAE) and its recall-correction factor 1/recall;
  * if NO model clears MIN_RECALL, the unit is marked ``fail`` and NO counts are
    emitted for it (raw AI counts are not trustworthy — hand-count or fine-tune);
  * otherwise run the recommended model on every Scene-1 frame of that unit in
    the manifest, count boxes at CONF, and record raw + recall-corrected counts.

Corrected count = raw / recall. The recall 95% CI (recall_lo, recall_hi from the
gate) gives a correction bracket: corrected_lo = raw / recall_hi,
corrected_hi = raw / recall_lo. This brackets ONLY detector-recall uncertainty;
frame-sampling scatter is reported separately as the weekly SEM. It is a
lower-confidence index than the manual box-counts, by construction.

Outputs (under notebooks/):
  ai_counts_gate_corrected_per_frame.csv   one row per scored frame
  worm_timeseries_gate_corrected.csv       weekly-Monday corrected mean/SEM (usable units)
  unit_gate_status.csv                      one row per unit (usable/fail + factors)

Env: CONF (0.25), MIN_RECALL (0.40), GATES_DIR, MANIFEST.
"""

from __future__ import annotations

import math
import os
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MANIFEST = Path(
    os.environ.get(
        "MANIFEST", str(REPO / "validation/monday_manual_series/frame_manifest.csv")
    )
)
GATES_DIR = Path(os.environ.get("GATES_DIR", str(REPO / "validation/unit_gates")))
IMG_DIR = REPO / "validation/monday_manual_series/images"
OUT_FRAME = REPO / "notebooks/ai_counts_gate_corrected_per_frame.csv"
OUT_SERIES = REPO / "notebooks/worm_timeseries_gate_corrected.csv"
OUT_STATUS = REPO / "notebooks/unit_gate_status.csv"
CONF = float(os.environ.get("CONF", "0.25"))
MIN_RECALL = float(os.environ.get("MIN_RECALL", "0.40"))
SWAP_MONTH = 8  # camera swapped each August -> unit Y spans Aug(Y)..Aug(Y+1)
MODEL_WEIGHTS = {
    "v2": REPO / "99_runs/scaleworm_v2/weights/best.pt",
    "v3": REPO / "99_runs/scaleworm_v3/weights/best.pt",
}


def unit_of(dt: datetime) -> str:
    """Camera unit for a datetime (August swap): CAMHDA301-Y, Y=year if month>=Aug else year-1."""
    y = dt.year if dt.month >= SWAP_MONTH else dt.year - 1
    return f"CAMHDA301-{y}"


def _extract(video: str, t: float, dest: Path) -> bool:
    """Extract the frame at ``t`` seconds from ``video`` into ``dest`` (cached PNG). True on success."""
    if not Path(video).exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
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
            str(dest),
        ],
        capture_output=True,
        check=False,
    )
    return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def recommend(gate: pd.DataFrame, min_recall: float = MIN_RECALL) -> dict | None:
    """Pick the gate-recommended model, mirroring score_unit_gate.py.

    Returns a dict with model/recall/recall_lo/recall_hi/correction, or None if
    no model clears ``min_recall`` (the unit fails the gate).
    """
    ok = gate[gate["recall"] >= min_recall]
    if ok.empty:
        return None
    best = ok.loc[ok["count_mae"].idxmin()]
    recall = float(best["recall"])
    return {
        "model": str(best["model"]),
        "recall": recall,
        "recall_lo": float(best["recall_lo"]),
        "recall_hi": float(best["recall_hi"]),
        "correction": 1.0 / recall,
    }


def correct(raw: int, rec: dict) -> tuple[float, float, float]:
    """Recall-correct a raw count; return (point, lo, hi) using the recall CI bracket."""
    point = raw / rec["recall"]
    lo = raw / rec["recall_hi"]  # higher recall -> less correction -> lower estimate
    hi = raw / rec["recall_lo"]
    return point, lo, hi


def weekly(frame_rows: list[dict]) -> list[dict]:
    """Aggregate corrected per-frame counts to a weekly-Monday mean/SEM per date.

    ``frame_rows`` need keys ``date`` (ISO) and ``corrected_count``.
    """
    by_day: dict[str, list[float]] = defaultdict(list)
    for r in frame_rows:
        by_day[r["date"]].append(float(r["corrected_count"]))
    out = []
    for d in sorted(by_day):
        c = by_day[d]
        n = len(c)
        sem = statistics.stdev(c) / math.sqrt(n) if n > 1 else float("nan")
        out.append(
            {
                "date": d,
                "n_slots": n,
                "mean_corrected": round(statistics.fmean(c), 4),
                "sem_corrected": round(sem, 4) if not math.isnan(sem) else "",
            }
        )
    return out


def load_unit_recs(
    gates_dir: Path, min_recall: float = MIN_RECALL
) -> dict[str, dict | None]:
    """Map unit -> recommendation dict (or None if it fails) for every gate_result.csv found."""
    recs: dict[str, dict | None] = {}
    for gr in sorted(gates_dir.glob("*/gate_result.csv")):
        recs[gr.parent.name] = recommend(pd.read_csv(gr), min_recall)
    return recs


def main() -> None:
    from ultralytics import YOLO

    recs = load_unit_recs(GATES_DIR)
    if not recs:
        raise SystemExit(f"no gate_result.csv under {GATES_DIR}")

    df = pd.read_csv(MANIFEST)
    models: dict[str, object] = {}
    frame_rows: list[dict] = []
    status: list[dict] = []

    for unit, rec in recs.items():
        sub = [
            r
            for _, r in df.iterrows()
            if unit_of(datetime.fromisoformat(r["datetime_utc"])) == unit
        ]
        base = {"unit": unit, "n_frames": len(sub)}
        if rec is None:
            status.append(
                {**base, "status": "fail", "model": "", "recall": "", "correction": ""}
            )
            print(
                f"{unit}: FAIL gate -> {len(sub)} frames NOT counted (hand-count/fine-tune)"
            )
            continue
        status.append(
            {
                **base,
                "status": "usable",
                "model": rec["model"],
                "recall": round(rec["recall"], 4),
                "correction": round(rec["correction"], 4),
            }
        )
        key = rec["model"]
        if key not in models:
            models[key] = YOLO(str(MODEL_WEIGHTS[key]))
        model = models[key]
        for r in sub:
            fid = r["frame_id"]
            img = IMG_DIR / f"{fid}.png"
            if not img.exists() and not _extract(
                r["video_path"], float(r["scene1_time_s"]), img
            ):
                print(f"  skip {fid}: extract failed / video missing")
                continue
            n = len(model.predict(str(img), conf=CONF, verbose=False)[0].boxes)
            point, lo, hi = correct(n, rec)
            frame_rows.append(
                {
                    "frame_id": fid,
                    "date": datetime.fromisoformat(r["datetime_utc"])
                    .date()
                    .isoformat(),
                    "unit": unit,
                    "model": key,
                    "raw_count": n,
                    "recall": round(rec["recall"], 4),
                    "corrected_count": round(point, 4),
                    "corrected_lo": round(lo, 4),
                    "corrected_hi": round(hi, 4),
                }
            )
        print(f"{unit}: {rec['model']} x{rec['correction']:.2f} on {len(sub)} frames")

    OUT_FRAME.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame_rows).to_csv(OUT_FRAME, index=False)
    pd.DataFrame(weekly(frame_rows)).to_csv(OUT_SERIES, index=False)
    pd.DataFrame(status).to_csv(OUT_STATUS, index=False)
    n_scored = len(frame_rows)
    n_usable = sum(s["status"] == "usable" for s in status)
    print(
        f"\nwrote {OUT_FRAME.name} ({n_scored} frames), {OUT_SERIES.name}, "
        f"{OUT_STATUS.name} ({n_usable}/{len(status)} units usable)"
    )


if __name__ == "__main__":
    main()
