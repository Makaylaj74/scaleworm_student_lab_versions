"""Re-count the clear-window harvest frames with v8_tiled + SAHI (the adopted clear detector).

The existing clear-window index (harvest_mf_2021_2023.csv) was counted with v5_all plain + a flat
/0.72 recall correction. v8_tiled+SAHI gates on clear at ~76% recall / 73.5% precision / 103% count
(near-unbiased) -- better recall AND no bias correction needed. This re-counts each clear-window
recording by REUSING its validated best_time_s pick (the frame-finder is ~90% right in the clear
window, so we keep the pick and only upgrade the COUNT), which is ~9x cheaper than re-running the
SAHI frame-finder.

Resumable (skips frame_ids already in --out). Set LIMIT>0 for a pilot.

Run (thesis venv: sahi + ultralytics + cuda):
  LIMIT=30 /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python scripts/recount_clear_v8sahi.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_frames import extract_frame, stem_to_video

REPO = Path("/home/jovyan/scaleworm-student-lab")
SAN = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
IN = REPO / "notebooks" / "harvest_mf_2021_2023.csv"
OUT = Path(os.environ.get("OUT", str(REPO / "notebooks" / "recount_clear_v8sahi.csv")))
MODEL = REPO / "99_runs/scaleworm_v8_tiled/weights/best.pt"
CONF = float(os.environ.get("CONF", "0.40"))
SLICE = int(os.environ.get("SLICE", "640"))
OVERLAP = float(os.environ.get("OVERLAP", "0.2"))
LIMIT = int(os.environ.get("LIMIT", "0"))
SCRATCH = Path("/tmp/claude-1000/-home-jovyan/d15fee80-de61-4938-b37b-519c259912ed/scratchpad/recount_frames")
CLEAR_LO, CLEAR_HI = "20210901", "20230811"
FIELDS = ["frame_id", "datetime_utc", "best_time_s", "count_v5_raw",
          "count_v5_corrected", "count_v8_sahi"]


def clear_rows() -> list[dict]:
    rows = []
    for r in csv.DictReader(IN.open()):
        day = r["frame_id"].split("-")[1][:8]
        if CLEAR_LO <= day < CLEAR_HI:
            rows.append(r)
    return rows


def done_ids() -> set[str]:
    if not OUT.exists():
        return set()
    return {r["frame_id"] for r in csv.DictReader(OUT.open())}


def main() -> None:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    SCRATCH.mkdir(parents=True, exist_ok=True)
    det = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(MODEL),
        confidence_threshold=CONF, device="cuda:0")

    rows = clear_rows()
    done = done_ids()
    todo = [r for r in rows if r["frame_id"] not in done]
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"clear-window rows={len(rows)}  done={len(done)}  this run={len(todo)}")

    write_header = not OUT.exists()
    f = OUT.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if write_header:
        w.writeheader()

    t0 = time.monotonic()
    n_ok = 0
    for i, r in enumerate(todo, 1):
        fid = r["frame_id"]
        vid = stem_to_video(fid, SAN)
        png = SCRATCH / f"{fid}.png"
        if not vid.exists() or not extract_frame(vid, float(r["best_time_s"]), png):
            continue
        pred = get_sliced_prediction(
            str(png), det, slice_height=SLICE, slice_width=SLICE,
            overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP, verbose=0)
        w.writerow({
            "frame_id": fid, "datetime_utc": r["datetime_utc"],
            "best_time_s": r["best_time_s"], "count_v5_raw": r["count_raw"],
            "count_v5_corrected": r["count_corrected"],
            "count_v8_sahi": len(pred.object_prediction_list),
        })
        png.unlink(missing_ok=True)
        n_ok += 1
        if i % 25 == 0:
            f.flush()
            rate = (time.monotonic() - t0) / i
            print(f"  {i}/{len(todo)}  {rate:.2f}s/rec  ETA {rate * (len(todo) - i) / 60:.0f}min")
    f.close()
    dt = time.monotonic() - t0
    print(f"done: {n_ok} recounted in {dt / 60:.1f}min ({dt / max(1, len(todo)):.2f}s/rec)")


if __name__ == "__main__":
    main()
