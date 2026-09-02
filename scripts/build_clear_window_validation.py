"""Build the CLEAR-WINDOW hand-count validation slice for scale-worm detection.

Purpose: a leakage-free ground-truth set to (1) measure the EXISTING detector's
recall in the clear image regime and (2) serve as a held-out test set for any
future retrain. Frames are front-on Mushroom ("Scene 1") views sampled across the
two clearest camera units:

  -2022 unit  (2023-01 .. 2023-07)  Scene-1 times already known from the sort log
                                    -> full-res frames extracted here, ready to count.
  -2021 unit  (2021-09 .. 2022-07)  no sort exists -> contact sheets rendered here;
                                    the counter records scene1_time_s, then runs
                                    extract_pending step to get the full-res frame.

Deterministic selection (no RNG). Output under validation/clear_window_handcount/.
Parallelism: 8 workers (Hub 24-cap; single agent).
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path("/home/jovyan/scaleworm-student-lab")
sys.path.insert(0, str(REPO / "scripts"))
from count_frames import extract_frame, stem_to_video
from scene_sampler import build_contact_sheet

VIDEO_ROOT = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
SORT_LOG = REPO / "scene_sorting/full_2023_2024/sort_log.csv"
WEEKLY = REPO / "notebooks/image_quality_weekly_2021_2023.csv"
OUT = REPO / "validation/clear_window_handcount"
FRAMES = OUT / "frames"
SHEETS = OUT / "contact_sheets"
SHEET_CSV = OUT / "handcount_sheet.csv"

PER_MONTH_2022 = 4  # -2022 unit, 2023-01..07  (Scene-1 known)
PER_MONTH_2021 = 2  # -2021 unit, 2021-09..2022-07 (needs contact sheet)
WORKERS = 8

COLS = [
    "frame_id",
    "datetime_utc",
    "camera_unit",
    "quarter",
    "sharpness_survey",
    "scene1_time_s",
    "frame_ready",
    "worm_count",
    "counter",
    "date_counted",
    "notes",
    "video_path",
]


def _even(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    idx = sorted(set(np.linspace(0, len(items) - 1, k).astype(int).tolist()))
    return [items[i] for i in idx]


def _quarter(dt: datetime) -> str:
    return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"


def select_2022() -> list[dict]:
    """4 scene1 recordings/month from 2023-01..07 (from the sort log)."""
    by_month: dict[str, list[tuple[str, str]]] = {}
    with SORT_LOG.open() as fh:
        for r in csv.DictReader(fh):
            if r["decision"] != "scene1":
                continue
            stem = r["stem"]
            ym = stem[10:16]  # YYYYMM
            if ym[:4] == "2023" and ym[4:6] in {
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
            }:
                by_month.setdefault(ym, []).append((stem, r["scene1_time_s"]))
    rows = []
    for ym in sorted(by_month):
        for stem, t in _even(sorted(by_month[ym]), PER_MONTH_2022):
            dt = datetime.strptime(stem.split("-")[1], "%Y%m%dT%H%M%S")  # noqa: DTZ007
            rows.append(
                {
                    "frame_id": stem,
                    "datetime_utc": dt.isoformat(),
                    "camera_unit": "CAMHDA301-2022",
                    "quarter": _quarter(dt),
                    "sharpness_survey": "",
                    "scene1_time_s": t,
                    "frame_ready": "",
                    "worm_count": "",
                    "counter": "",
                    "date_counted": "",
                    "notes": "",
                    "video_path": str(stem_to_video(stem, VIDEO_ROOT)),
                }
            )
    return rows


def select_2021() -> list[dict]:
    """2 recordings/month from 2021-09..2022-07 (from the weekly survey)."""
    # naive UTC on purpose: filename timestamps are UTC, compared consistently
    lo, hi = datetime(2021, 9, 1), datetime(2022, 8, 1)  # noqa: DTZ001
    by_month: dict[str, list[tuple[str, str, str]]] = {}
    with WEEKLY.open() as fh:
        for r in csv.DictReader(fh):
            if not r["n_frames"] or int(r["n_frames"]) < 4:
                continue
            dt = datetime.strptime(r["ts"], "%Y%m%dT%H%M%S")  # noqa: DTZ007
            if not (lo <= dt < hi):
                continue
            ym = dt.strftime("%Y%m")
            by_month.setdefault(ym, []).append((r["ts"], r["path"], r["sharp_median"]))
    rows = []
    for ym in sorted(by_month):
        for ts, path, sharp in _even(sorted(by_month[ym]), PER_MONTH_2021):
            dt = datetime.strptime(ts, "%Y%m%dT%H%M%S")  # noqa: DTZ007
            stem = f"CAMHDA301-{ts}"
            rows.append(
                {
                    "frame_id": stem,
                    "datetime_utc": dt.isoformat(),
                    "camera_unit": "CAMHDA301-2021",
                    "quarter": _quarter(dt),
                    "sharpness_survey": sharp,
                    "scene1_time_s": "",
                    "frame_ready": "pending-contact-sheet",
                    "worm_count": "",
                    "counter": "",
                    "date_counted": "",
                    "notes": "",
                    "video_path": path,
                }
            )
    return rows


def main() -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    SHEETS.mkdir(parents=True, exist_ok=True)
    rows = select_2022() + select_2021()
    rows.sort(key=lambda r: r["datetime_utc"])

    # 1) extract full-res Scene-1 frames for -2022 rows (Scene-1 time known)
    ready = [r for r in rows if r["scene1_time_s"]]
    print(
        f"extracting {len(ready)} full-res Scene-1 frames (-2022 unit)...", flush=True
    )
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {}
        for r in ready:
            out = FRAMES / f"{r['frame_id']}.png"
            futs[
                ex.submit(
                    extract_frame, Path(r["video_path"]), float(r["scene1_time_s"]), out
                )
            ] = (r, out)
        for fut in as_completed(futs):
            r, out = futs[fut]
            r["frame_ready"] = (
                "Y" if (fut.result() and out.exists()) else "extract-failed"
            )

    # 2) contact sheets for -2021 rows (counter picks Scene-1 from these)
    pend = [r for r in rows if not r["scene1_time_s"]]
    print(f"building {len(pend)} contact sheets (-2021 unit)...", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(
                build_contact_sheet,
                Path(r["video_path"]),
                SHEETS / f"{r['frame_id']}.png",
            ): r
            for r in pend
        }
        for fut in as_completed(futs):
            r = futs[fut]
            r["notes"] = (
                "contact sheet ready" if fut.result() else "contact-sheet-failed"
            )

    with SHEET_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    n22 = sum(r["camera_unit"].endswith("2022") for r in rows)
    n21 = len(rows) - n22
    ready_n = sum(r["frame_ready"] == "Y" for r in rows)
    print(f"\nDONE: {len(rows)} recordings  (-2022: {n22}, -2021: {n21})")
    print(f"  ready-to-count frames extracted: {ready_n}")
    print(f"  contact sheets for scene-picking: {len(pend)}")
    print(f"sheet -> {SHEET_CSV}")


if __name__ == "__main__":
    main()
