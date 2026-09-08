"""Build contact sheets for a -2021 TRAINING pick set (v2 retrain, Phase 1).

Selects ~40 CAMHDA301-2021 recordings (2021-09..2022-07) from the weekly image-quality
survey, EXCLUDING the 22 recordings already used in the validation set (no leakage), and
renders a labeled contact sheet per recording. Makayla then picks the front-on Scene-1
time for each in notebooks/32_pick_scene1_2021_train.ipynb; make_prelabels.py extracts +
pre-labels the picked frames for correction.

    python scripts/build_train_2021_contact_sheets.py
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from scene_sampler import build_contact_sheet

REPO = Path("/home/jovyan/scaleworm-student-lab")
WEEKLY = REPO / "notebooks/image_quality_weekly_2021_2023.csv"
VAL_SHEET = REPO / "validation/clear_window_handcount/handcount_sheet.csv"
OUT = REPO / "datasets/scaleworm_v2/train_pick_2021"
SHEETS = OUT / "contact_sheets"
PICK_CSV = OUT / "pick_sheet.csv"

PER_MONTH = 4  # ~4/month x 11 months -> ~44, capped to TARGET
TARGET = 40
WORKERS = 8  # well under the 24-worker cap


def even(items: list, k: int) -> list:
    """Evenly spaced k picks from a sorted list (deterministic)."""
    if k >= len(items):
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def select() -> list[dict]:
    val_stems = {
        r["frame_id"]
        for r in csv.DictReader(VAL_SHEET.open())
        if r["camera_unit"] == "CAMHDA301-2021"
    }
    lo, hi = datetime(2021, 9, 1), datetime(2022, 8, 1)  # noqa: DTZ001
    by_month: dict[str, list] = {}
    for r in csv.DictReader(WEEKLY.open()):
        if not r["n_frames"] or int(r["n_frames"]) < 4:
            continue
        dt = datetime.strptime(r["ts"], "%Y%m%dT%H%M%S")  # noqa: DTZ007
        if not (lo <= dt < hi):
            continue
        stem = f"CAMHDA301-{r['ts']}"
        if stem in val_stems:
            continue  # leakage guard
        by_month.setdefault(dt.strftime("%Y%m"), []).append((r["ts"], r["path"], r["sharp_median"]))

    rows = []
    for ym in sorted(by_month):
        for ts, path, sharp in even(sorted(by_month[ym]), PER_MONTH):
            dt = datetime.strptime(ts, "%Y%m%dT%H%M%S")  # noqa: DTZ007
            rows.append({
                "frame_id": f"CAMHDA301-{ts}",
                "datetime_utc": dt.isoformat(),
                "camera_unit": "CAMHDA301-2021",
                "sharpness_survey": sharp,
                "scene1_time_s": "",
                "frame_ready": "pending-contact-sheet",
                "notes": "",
                "video_path": path,
            })
    return rows[:TARGET]


def main() -> None:
    SHEETS.mkdir(parents=True, exist_ok=True)
    rows = select()
    with PICK_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"building {len(rows)} contact sheets ({WORKERS} workers)...", flush=True)
    ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(build_contact_sheet, Path(r["video_path"]), SHEETS / f"{r['frame_id']}.png"): r
            for r in rows
            if not (SHEETS / f"{r['frame_id']}.png").exists()  # resumable: skip built sheets
        }
        for fut in as_completed(futs):
            ok += 1 if fut.result() else 0
    print(f"done: {ok}/{len(rows)} sheets -> {SHEETS}")
    print(f"pick sheet -> {PICK_CSV}")
    print("NEXT: notebooks/32_pick_scene1_2021_train.ipynb -> pick Scene-1 times")


if __name__ == "__main__":
    main()
