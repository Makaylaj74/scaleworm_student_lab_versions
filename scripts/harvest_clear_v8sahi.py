"""Re-harvest the clear window (2021-2023) with the VALIDATED recipe: v8_tiled + SAHI, counting
the MAX over a coarse 60 s temporal grid, no /0.72 correction.

validate_frame_finder.py showed this recipe reaches ~93% of manual Scene-1 abundance (vs the old
v5 + med-top3 + /0.72 harvest at ~58%) -- the gain is the better counter + max-over-grid aggregation,
not a new frame-finder. This produces the improved automated clear-window index.

For each clear-window recording (from the existing manifest): extract the 60 s grid frames (parallel
ffmpeg), run v8+SAHI on each, record the MAX count and the time it occurred. Resumable (skips
frame_ids already in --out). NOTE: this is a RELATIVE index -- 93% pooled is a cancellation of
sparse-frame max-bias (over) and dense-peak density-ceiling (under); residual is the documented ceiling.

Run (thesis venv: sahi + cuda):
  /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python scripts/harvest_clear_v8sahi.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_frames import extract_frame, stem_to_video

REPO = Path("/home/jovyan/scaleworm-student-lab")
SAN = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
IN = REPO / "notebooks" / "harvest_mf_2021_2023.csv"
OUT = Path(os.environ.get("OUT", str(REPO / "notebooks" / "harvest_clear_v8sahi_2021_2023.csv")))
MODEL = REPO / "99_runs/scaleworm_v8_tiled/weights/best.pt"
CONF = float(os.environ.get("CONF", "0.40"))
SLICE = int(os.environ.get("SLICE", "640"))
OVERLAP = float(os.environ.get("OVERLAP", "0.2"))
GRID = list(range(int(os.environ.get("T0", "90")), int(os.environ.get("T1", "630")) + 1,
                  int(os.environ.get("STEP", "60"))))  # validated coarse grid
EXT_WORKERS = int(os.environ.get("EXT_WORKERS", "8"))  # ffmpeg threads (lab cap: <=24)
LIMIT = int(os.environ.get("LIMIT", "0"))
SCRATCH = Path("/tmp/claude-1000/-home-jovyan/d15fee80-de61-4938-b37b-519c259912ed/scratchpad/harvest_v8")
CLEAR_LO, CLEAR_HI = "20210901", "20230811"
FIELDS = ["frame_id", "datetime_utc", "best_time_s", "count_v8sahi_max", "n_frames", "all_counts"]


def clear_rows() -> list[dict]:
    return [r for r in csv.DictReader(IN.open())
            if CLEAR_LO <= r["frame_id"].split("-")[1][:8] < CLEAR_HI]


def done_ids() -> set[str]:
    return {r["frame_id"] for r in csv.DictReader(OUT.open())} if OUT.exists() else set()


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
    print(f"clear rows={len(rows)} done={len(done)} this run={len(todo)} grid={GRID}")

    write_header = not OUT.exists()
    fout = OUT.open("a", newline="")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    if write_header:
        w.writeheader()

    def sahi_count(png: Path) -> int:
        r = get_sliced_prediction(
            str(png), det, slice_height=SLICE, slice_width=SLICE,
            overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP, verbose=0)
        return len(r.object_prediction_list)

    t0 = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=EXT_WORKERS)
    for i, r in enumerate(todo, 1):
        fid = r["frame_id"]
        vid = stem_to_video(fid, SAN)
        if not vid.exists():
            continue
        # parallel extraction of the grid frames
        def _ex(t, fid=fid, vid=vid):
            png = SCRATCH / f"{fid}_{t}.png"
            return (t, png) if extract_frame(vid, float(t), png) else (t, None)
        extracted = list(pool.map(_ex, GRID))
        counts = {}
        for t, png in extracted:
            if png is None:
                continue
            counts[t] = sahi_count(png)
            png.unlink(missing_ok=True)
        if not counts:
            continue
        best_t = max(counts, key=counts.get)
        w.writerow({
            "frame_id": fid, "datetime_utc": r["datetime_utc"], "best_time_s": best_t,
            "count_v8sahi_max": counts[best_t], "n_frames": len(counts),
            "all_counts": ";".join(f"{t}:{counts[t]}" for t in sorted(counts)),
        })
        if i % 10 == 0:
            fout.flush()
            rate = (time.monotonic() - t0) / i
            print(f"  {i}/{len(todo)}  {rate:.1f}s/rec  ETA {rate * (len(todo) - i) / 3600:.1f}h", flush=True)
    pool.shutdown()
    fout.close()
    dt = time.monotonic() - t0
    print(f"done: {len(todo)} recs in {dt / 3600:.2f}h ({dt / max(1, len(todo)):.1f}s/rec)")


if __name__ == "__main__":
    main()
