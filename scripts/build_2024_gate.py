"""Scaffold a 2024 blurry hand-count validation gate (detector-chosen frames).

Picks ~30 2024 Mon-Fri recordings (monthly spread), scans the pan window to find each
recording's max-count (front-on) frame at the harvest's operating point (v5_all, conf 0.40),
and SAVES that frame. MJ then click-counts those exact frames (nb29), so the gate measures
the detector's count against human truth on the SAME frame the harvest uses.

Sequential ffmpeg (1 worker) so it is safe alongside the running 16-worker harvest (<=24 cap).

Run: python scripts/build_2024_gate.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from count_frames import count_worms, extract_frame

MANIFEST = Path("/tmp/claude-1000/-home-jovyan/8dad8b25-80ff-4baa-bbbc-40714f027bd6/scratchpad/y2024_manifest.csv")
MODEL = REPO / "99_runs/scaleworm_v5_all/weights/best.pt"
OUT = REPO / "validation/blurry_handcount_2024"
CONF = 0.40
T0, T1, STEP = 120, 600, 60
PER_MONTH = 3


def main() -> None:
    from ultralytics import YOLO

    rows = list(csv.DictReader(MANIFEST.open()))
    by_month: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_month[r["datetime_utc"][:7]].append(r)
    picks = []
    for mo in sorted(by_month):
        month = sorted(by_month[mo], key=lambda r: r["frame_id"])
        step = max(1, len(month) // PER_MONTH)
        picks += month[::step][:PER_MONTH]
    print(f"selected {len(picks)} 2024 recordings across {len(by_month)} months")

    model = YOLO(str(MODEL))
    names = model.names
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    (OUT / "clicks").mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    times = [T0 + i * STEP for i in range(int((T1 - T0) / STEP) + 1)]
    sheet = []
    for k, r in enumerate(picks):
        best_c, best_t = -1, None
        for t in times:
            png = tmp / "f.png"
            if not extract_frame(Path(r["video_path"]), t, png):
                continue
            res = model(str(png), conf=CONF, verbose=False)[0]
            c = count_worms([int(x) for x in res.boxes.cls.tolist()], names)
            if c > best_c:
                best_c, best_t = c, t
        if best_t is None:
            continue
        fid = r["frame_id"]
        extract_frame(Path(r["video_path"]), best_t, OUT / "frames" / f"{fid}.png")
        sheet.append({
            "frame_id": fid,
            "datetime_utc": r["datetime_utc"],
            "camera_unit": "CAMHDA301-2024",
            "quarter": f"2024-Q{(int(r['datetime_utc'][5:7]) - 1) // 3 + 1}",
            "sharpness_survey": "",
            "scene1_time_s": best_t,
            "frame_ready": "Y",
            "worm_count": "",
            "counter": "",
            "date_counted": "",
            "notes": "detector-chosen frame (max-count @conf0.40); blurry-2024 gate",
            "video_path": r["video_path"],
            "detector_count": best_c,  # what the detector counted at this frame
        })
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(picks)}")
    fields = list(sheet[0].keys())
    with (OUT / "handcount_sheet.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sheet)
    (OUT / "README.md").write_text(
        "*AI-generated draft (Claude, Anthropic) — for review. Frame selection is "
        "version-controlled (scripts/build_2024_gate.py); worm counts entered here are the "
        "counter's own data.*\n\n"
        "# 2024 blurry hand-count validation gate\n\n"
        "~30 detector-chosen (max-count @conf0.40) 2024 Scene-1 frames for validating the "
        "blurry-2024 harvest. Click-count in nb29 (set BASE to this folder). Then compare "
        "worm_count vs the `detector_count` column to get 2024 blurry recall.\n"
    )
    print(f"wrote {OUT}/handcount_sheet.csv ({len(sheet)} frames) + frames/ + README")


if __name__ == "__main__":
    main()
