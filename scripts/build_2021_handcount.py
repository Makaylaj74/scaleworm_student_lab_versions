"""Scaffold a Jan-Aug 2021 hand-count gate to verify the pre-Sep-2021 detector undercount.

The per-quarter automated-vs-manual check showed the detector recovers only ~50-58% of manual
worms in late 2021 (vs ~100% in 2022-2023), but the manual Monday series starts 2021-09-06, so
Jan-Aug 2021 has NO ground truth. This picks ~2 recordings/month across Jan-Aug 2021, extracts
each at the harvest's already-chosen front-on time (best_time_s), re-counts that exact frame with
the deployed detector (v5_all @ conf 0.40 = the harvest operating point), and saves it. MJ then
click-counts the SAME frames (nb29) so recall = detector_count / MJ_clicks confirms (or refutes)
that the ~2x undercount extends before September.

Independent click GT (not box-correction) -> valid recall AND precision, like clear_window_handcount.
Sequential ffmpeg (1 worker), well under the 24-core cap.

Run: /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/build_2021_handcount.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from count_frames import count_worms, extract_frame

HARVEST = REPO / "notebooks/harvest_mf_2021_2023.csv"
MODEL = REPO / "99_runs/scaleworm_v5_all/weights/best.pt"
SAN = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
OUT = REPO / "validation/early_2021_handcount"
CONF = 0.40
PER_MONTH = 2
START, END = "2021-01-01", "2021-09-01"


def video_path(frame_id: str) -> Path:
    """.../YYYY/MM/DD/<frame_id>.mp4 (compressed proxy the harvest used)."""
    d = frame_id.split("-")[1][:8]  # YYYYMMDD
    return SAN / d[:4] / d[4:6] / d[6:8] / f"{frame_id}.mp4"


def main() -> None:
    import pandas as pd
    from ultralytics import YOLO

    h = pd.read_csv(HARVEST)
    h["dt"] = pd.to_datetime(h["datetime_utc"]).dt.tz_localize(None)
    e = h[(h["dt"] >= START) & (h["dt"] < END) & (h["count_raw"] > 0)].copy()

    by_month: dict[str, list] = defaultdict(list)
    for _, r in e.iterrows():
        by_month[r["dt"].strftime("%Y-%m")].append(r)
    picks = []
    for mo in sorted(by_month):
        month = sorted(by_month[mo], key=lambda r: r["frame_id"])
        step = max(1, len(month) // PER_MONTH)
        picks += month[::step][:PER_MONTH]
    print(f"selected {len(picks)} recordings across {len(by_month)} months (Jan-Aug 2021)")

    model = YOLO(str(MODEL))
    names = model.names
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    (OUT / "clicks").mkdir(exist_ok=True)

    sheet = []
    for k, r in enumerate(picks):
        fid = r["frame_id"]
        vp = video_path(fid)
        if not vp.exists():
            print(f"  MISSING video: {vp.name}")
            continue
        t = int(r["best_time_s"])  # harvest's front-on pick
        out_png = OUT / "frames" / f"{fid}.png"
        if not extract_frame(vp, t, out_png):
            print(f"  extract failed: {fid} @ {t}s")
            continue
        res = model(str(out_png), conf=CONF, verbose=False)[0]
        det = count_worms([int(x) for x in res.boxes.cls.tolist()], names)
        sheet.append({
            "frame_id": fid,
            "datetime_utc": r["datetime_utc"],
            "camera_unit": "CAMHDA301-2021",
            "quarter": f"2021-Q{(r['dt'].month - 1) // 3 + 1}",
            "scene1_time_s": t,
            "frame_ready": "Y",
            "worm_count": "",
            "counter": "",
            "date_counted": "",
            "notes": "harvest front-on frame (best_time_s); early-2021 undercount gate",
            "video_path": str(vp),
            "detector_count": det,              # detector count on THIS exact frame
            "harvest_corrected": round(float(r["count_corrected"]), 1),  # harvest's ÷0.72 value
        })
        if (k + 1) % 5 == 0:
            print(f"  {k + 1}/{len(picks)}")

    fields = list(sheet[0].keys())
    with (OUT / "handcount_sheet.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sheet)
    (OUT / "README.md").write_text(
        "*AI-generated draft (Claude, Anthropic) — for review. Frame selection is "
        "version-controlled (scripts/build_2021_handcount.py); worm counts entered here are the "
        "counter's own data.*\n\n"
        "# Jan-Aug 2021 hand-count gate (pre-Sep-2021 undercount check)\n\n"
        f"{len(sheet)} harvest-chosen front-on 2021 Scene-1 frames (≈2/month, Jan-Aug). Click-count "
        "in nb29 (set BASE to this folder). Then compare `worm_count` (your clicks = truth) vs "
        "`detector_count` (detector on the same frame): recall = detector/clicks. Confirms whether "
        "the ~50-58% undercount measured for Sep-Dec 2021 (vs manual) extends before September. "
        "`harvest_corrected` is the harvest's ÷0.72 value for reference.\n"
    )
    det_tot = sum(s["detector_count"] for s in sheet)
    print(f"wrote {OUT}/handcount_sheet.csv ({len(sheet)} frames) + frames/ + clicks/ + README")
    print(f"detector total on these frames: {det_tot} ({det_tot / len(sheet):.1f}/frame) — awaiting MJ clicks")


if __name__ == "__main__":
    main()
