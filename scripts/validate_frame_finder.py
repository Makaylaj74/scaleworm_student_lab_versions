"""Validate whether a better FRAME-FINDER can close the automated-index gap.

The re-count showed the auto index sits at ~58-60% of manual Scene-1 truth because the auto-picked
frame isn't the true front-on moment -- the bottleneck is frame-finding, not counting. This tests, on
Monday frames where MJ marked the TRUE Scene-1 time, whether a finer temporal search with v8+SAHI can
recover the manual count (i.e. is frame-finding winnable).

For each sampled recording, v8+SAHI is run at:
  * MJ's scene1_time_s         -> v8_at_manual   (counting ceiling on the true frame)
  * a FINE grid (STEP s)       -> fine_max       (dense auto-finder's best frame)
  * a COARSE 60 s subset       -> coarse_max      (current harvest finder, as a baseline)
Manual truth = box count in the Monday label file.

Reports, pooled: fine_auto/manual, coarse_auto/manual, v8_at_manual/manual, and the localization
error |fine_max_time - scene1_time|. If fine_auto/manual -> ~1 and localization is tight, frame-finding
is tractable (major result); if not, front-on localization on pan footage is fundamentally hard.

Run (thesis venv: sahi + cuda):
  LIMIT=24 STEP=20 /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python scripts/validate_frame_finder.py
"""

from __future__ import annotations

import csv
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_frames import extract_frame, stem_to_video

REPO = Path("/home/jovyan/scaleworm-student-lab")
SAN = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
MON = REPO / "validation" / "monday_manual_series"
MODEL = REPO / "99_runs/scaleworm_v8_tiled/weights/best.pt"
CONF = float(os.environ.get("CONF", "0.40"))
SLICE = int(os.environ.get("SLICE", "640"))
OVERLAP = float(os.environ.get("OVERLAP", "0.2"))
STEP = int(os.environ.get("STEP", "20"))
T0, T1 = int(os.environ.get("T0", "90")), int(os.environ.get("T1", "660"))
LIMIT = int(os.environ.get("LIMIT", "24"))
SCRATCH = Path("/tmp/claude-1000/-home-jovyan/d15fee80-de61-4938-b37b-519c259912ed/scratchpad/ff_val")


def sample() -> list[dict]:
    """Even-spread clear-window Monday rows that have scene1_time_s + a label file."""
    rows = []
    for r in csv.DictReader((MON / "frame_manifest.csv").open()):
        day = r["frame_id"].split("-")[1][:8]
        lbl = MON / "labels" / f"{r['frame_id']}.txt"
        if "20210901" <= day < "20230811" and r.get("scene1_time_s") and lbl.exists():
            r["_manual"] = sum(1 for ln in lbl.read_text().splitlines() if ln.strip())
            rows.append(r)
    rows.sort(key=lambda r: r["frame_id"])
    if LIMIT and len(rows) > LIMIT:
        stepi = len(rows) / LIMIT
        rows = [rows[int(i * stepi)] for i in range(LIMIT)]
    return rows


def main() -> None:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    SCRATCH.mkdir(parents=True, exist_ok=True)
    det = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(MODEL),
        confidence_threshold=CONF, device="cuda:0")

    def count_at(vid: Path, t: float, tag: str) -> int | None:
        png = SCRATCH / f"{tag}.png"
        if not extract_frame(vid, t, png):
            return None
        r = get_sliced_prediction(
            str(png), det, slice_height=SLICE, slice_width=SLICE,
            overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP, verbose=0)
        png.unlink(missing_ok=True)
        return len(r.object_prediction_list)

    grid = list(range(T0, T1 + 1, STEP))
    recs = []
    for r in sample():
        fid = r["frame_id"]
        vid = stem_to_video(fid, SAN)
        if not vid.exists():
            continue
        man_t = float(r["scene1_time_s"])
        manual = r["_manual"]
        at_manual = count_at(vid, man_t, f"{fid}_man")
        counts = {t: count_at(vid, t, f"{fid}_{t}") for t in grid}
        counts = {t: c for t, c in counts.items() if c is not None}
        if not counts or at_manual is None:
            continue
        fine_t = max(counts, key=counts.get)
        fine_c = counts[fine_t]
        coarse = {t: c for t, c in counts.items() if (t - T0) % 60 == 0}
        coarse_c = max(coarse.values()) if coarse else fine_c
        recs.append({"manual": manual, "at_manual": at_manual, "fine": fine_c,
                     "coarse": coarse_c, "loc_err": abs(fine_t - man_t)})
        print(f"  {fid}: manual={manual:2d} v8@manual={at_manual:2d} "
              f"fine_max={fine_c:2d}@{fine_t}s coarse_max={coarse_c:2d} "
              f"locErr={abs(fine_t - man_t):.0f}s")

    def ratio(key):
        num = sum(r[key] for r in recs)
        den = sum(r["manual"] for r in recs)
        return num / den if den else 0.0

    print(f"\n=== FRAME-FINDER VALIDATION  n={len(recs)}  v8+SAHI  grid {T0}-{T1}/{STEP}s ===")
    print(f"  manual total/mean      : {sum(r['manual'] for r in recs)} / "
          f"{st.mean([r['manual'] for r in recs]):.1f}/fr")
    print(f"  v8 @ MJ's Scene-1 time : {ratio('at_manual'):.1%} of manual  (counting ceiling)")
    print(f"  FINE auto-finder       : {ratio('fine'):.1%} of manual  (dense search)")
    print(f"  COARSE 60s finder      : {ratio('coarse'):.1%} of manual  (current method)")
    print(f"  localization |fine-MJ| : median {st.median([r['loc_err'] for r in recs]):.0f}s  "
          f"mean {st.mean([r['loc_err'] for r in recs]):.0f}s")


if __name__ == "__main__":
    main()
