"""Build the unified manual-series frame manifest for scaleworm annotation.

One row per Scene-1 frame, drawn from one or more Scene-1 ``sort_log.csv`` files
(rows where ``decision == scene1``). The manifest is the index for the unified
box-correction workflow: a single annotation pass fills the boxes, from which the
manual worm timeseries (count = number of boxes) and the detector training labels
are both derived. See ``notebooks/manual_series_schema.md``.

Re-running is safe: if the output manifest already exists, existing annotation
columns are preserved per ``frame_id`` and only new frames are appended — so a
later run that adds the 2021-2022 sort will not clobber 2023-2024 counts.

Usage:
    python scripts/build_frame_manifest.py \
        scene_sorting/full_2023_2024/sort_log.csv \
        [scene_sorting/clear_window_2021_2022/sort_log.csv ...] \
        --out validation/monday_manual_series/frame_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

VIDEO_ROOT = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")

# Columns regenerated from source data on every build.
SOURCE_COLS = [
    "frame_id",
    "datetime_utc",
    "camera_unit",
    "quarter",
    "scene1_time_s",
    "video_path",
    "sharpness",
    "split",
]
# Annotation columns owned by the human/annotation tool; preserved across rebuilds.
ANNOTATION_COLS = [
    "frame_status",
    "worm_count",
    "label_path",
    "annotation_method",
    "prelabel_model",
    "counter",
    "date_counted",
    "notes",
]
COLUMNS = SOURCE_COLS + ANNOTATION_COLS

# Every 5th Monday is held out for model validation. Anchored to the date's
# ordinal week so the assignment is STABLE per date — independent of which other
# frames are in the build, so appending more years never reshuffles the split.
VAL_EVERY = 5


def stem_to_dt(stem: str) -> datetime:
    """Parse a CAMHD stem like ``CAMHDA301-20230102T001500`` to a datetime."""
    ts = stem.split("-", 1)[1]
    return datetime.strptime(ts, "%Y%m%dT%H%M%S")


def quarter(dt: datetime) -> str:
    """Return the calendar quarter label, e.g. ``2023-Q1``."""
    return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"


def split_for(dt: datetime) -> str:
    """Assign train/val by whole day (all slots of a day share a split).

    Deterministic and stable per date: every ``VAL_EVERY``-th day-block is val.
    """
    week_index = dt.date().toordinal() // 7
    return "val" if week_index % VAL_EVERY == 0 else "train"


def video_path(stem: str, dt: datetime) -> str:
    """Construct the on-disk .mp4 path for a stem."""
    return str(VIDEO_ROOT / f"{dt.year}" / f"{dt.month:02d}" / f"{dt.day:02d}" / f"{stem}.mp4")


def load_sharpness(paths: list[Path]) -> dict[str, str]:
    """Map frame_id -> sharpness from image-quality CSV(s) (stem, sharpness cols)."""
    out: dict[str, str] = {}
    for p in paths:
        if not p.exists():
            continue
        with p.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("stem") and row.get("sharpness"):
                    out[row["stem"]] = row["sharpness"]
    return out


def read_scene1_stems(sort_logs: list[Path]) -> list[tuple[str, str]]:
    """Return (stem, scene1_time_s) for every decision==scene1 row across logs."""
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for log in sort_logs:
        with log.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("decision") != "scene1":
                    continue
                stem = row["stem"]
                if stem in seen:  # first log wins on duplicates
                    continue
                seen.add(stem)
                rows.append((stem, row.get("scene1_time_s", "")))
    return rows


def load_existing(out_path: Path) -> dict[str, dict[str, str]]:
    """Load existing manifest keyed by frame_id (to preserve annotations)."""
    if not out_path.exists():
        return {}
    with out_path.open(newline="") as fh:
        return {r["frame_id"]: r for r in csv.DictReader(fh)}


def build_row(
    stem: str,
    scene1_time_s: str,
    sharpness: dict[str, str],
    prior: dict[str, str] | None,
) -> dict[str, str]:
    dt = stem_to_dt(stem)
    row = {c: "" for c in COLUMNS}
    row.update(
        frame_id=stem,
        datetime_utc=dt.isoformat(),
        camera_unit=f"CAMHDA301-{dt.year}",
        quarter=quarter(dt),
        scene1_time_s=scene1_time_s,
        video_path=video_path(stem, dt),
        sharpness=sharpness.get(stem, ""),
        split=split_for(dt),
    )
    if prior:  # preserve human-owned columns across rebuilds
        for c in ANNOTATION_COLS:
            row[c] = prior.get(c, "")
    else:
        row["frame_status"] = "pending"
    return row


def build(sort_logs: list[Path], sharpness_csvs: list[Path], out_path: Path) -> list[dict]:
    sharpness = load_sharpness(sharpness_csvs)
    prior = load_existing(out_path)
    rows = [
        build_row(stem, t, sharpness, prior.get(stem))
        for stem, t in read_scene1_stems(sort_logs)
    ]
    rows.sort(key=lambda r: r["datetime_utc"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sort_logs", nargs="+", type=Path, help="Scene-1 sort_log.csv file(s)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/monday_manual_series/frame_manifest.csv"),
    )
    ap.add_argument(
        "--sharpness",
        type=Path,
        nargs="*",
        default=[Path("notebooks/image_quality_timeseries.csv")],
        help="image-quality CSV(s) with stem,sharpness columns",
    )
    args = ap.parse_args()
    rows = build(args.sort_logs, args.sharpness, args.out)
    n_val = sum(r["split"] == "val" for r in rows)
    n_missing = sum(not Path(r["video_path"]).exists() for r in rows)
    print(f"wrote {len(rows)} Scene-1 rows -> {args.out}")
    print(f"  split: {len(rows) - n_val} train / {n_val} val")
    print(f"  videos missing on disk: {n_missing}")


if __name__ == "__main__":
    main()
