"""Scaffold the 2015-2016 early-recovery hand-count gate (post-Axial-eruption era).

Selects a monthly spread of confirmed Scene-1 recordings (2015 from full_2015 sort,
2016 from the subsample_2016 sort), extracts the full-res Scene-1 frame for each, and
writes a blank hand-count sheet + clicks/ dir + README — mirroring the clear_window
handcount layout so the nb29 click-counter works by just repointing its BASE.

IMPORTANT (No Borrowed Assumptions): this only stages FRAMES. The count TARGET for the
2015-2016 dense-mound/tubeworm scene is NOT the 2021-2024 mushroom-tower rubric and is
NOT yet defined -- that is an advisor call (eruption/colonization context). Do NOT count
until the target is decided; the README says so. The gate exists to (later) measure the
v5 detector's real recall on this era and/or to seed a v6 fine-tune.

Run: python scripts/build_early_recovery_gate.py   (add --dry-run to preview selection)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = {
    "2015": {
        "sort": REPO / "scene_sorting/full_2015/sort_log.csv",
        "manifest": REPO / "scene_sorting/full_2015/manifest.csv",
        "per_month": 2,  # 120 scene1 available -> even monthly spread
    },
    "2016": {
        "sort": REPO / "scene_sorting/subsample_2016/sort_log.csv",
        "manifest": REPO / "scene_sorting/subsample_2016/manifest.csv",
        "per_month": 99,  # only ~18 confirmed scene1 -> take all
    },
}
OUT = REPO / "validation/early_recovery_handcount_2015_2016"


def yyyymm(stem: str) -> str:
    return stem.split("-")[1][:6]


def even_pick(items: list, k: int) -> list:
    """Evenly-spaced deterministic pick of k items from a sorted list."""
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def select() -> list[dict]:
    picks = []
    for year, cfg in SRC.items():
        vids = {r["stem"]: r["video_path"] for r in csv.DictReader(cfg["manifest"].open())}
        times = {}
        by_month: dict[str, list[str]] = defaultdict(list)
        for r in csv.DictReader(cfg["sort"].open()):
            if r.get("decision") != "scene1":
                continue
            times[r["stem"]] = r.get("scene1_time_s") or ""
            by_month[yyyymm(r["stem"])].append(r["stem"])
        for month in sorted(by_month):
            for stem in even_pick(sorted(by_month[month]), cfg["per_month"]):
                picks.append(
                    {
                        "frame_id": stem,
                        "year": year,
                        "month": month,
                        "scene1_time_s": times.get(stem, ""),
                        "video_path": vids.get(stem, ""),
                    }
                )
    return picks


def extract(video: str, t: float, dst: Path) -> bool:
    cmd = ["ffmpeg", "-y", "-ss", str(t), "-i", video,
           "-frames:v", "1", "-q:v", "2", str(dst)]
    r = subprocess.run(cmd, capture_output=True, check=False)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


README = """*AI-generated draft (Claude, Anthropic) — for review. The frame selection and \
extraction are version-controlled (scripts/build_early_recovery_gate.py); any worm counts \
entered here are the counter's own data.*

# Early-recovery hand-count gate (2015-2016)

Held-out hand-count set for the **post-Axial-eruption early-recovery era** (the 2015
eruption is the scientific motivation for extending the scaleworm record backward). Its
purpose is to measure the **real** recall/precision of the detector on this era and/or to
seed a fine-tune (v6) — exactly the role `clear_window_handcount` played for 2021-2024.

## ⚠️ Do not count yet — the count TARGET is undefined

The 2015-2016 view is a **dense tubeworm/community mound**, visually different from the
2021-2024 mushroom-tower scene the detector was trained on. The 2021-2024 count rubric
does **not** transfer (No Borrowed Assumptions). Before anyone clicks:

1. Decide **what a countable unit is** in this scene (individual scale worms? occupied
   tubeworm patches? % cover?) **with the advisor**, using the eruption/colonization
   context (was Mushroom physically disturbed in April 2015? what is the succession
   target?).
2. Record that definition here, then count.

## Provenance

- Frames: Scene-1 recordings confirmed by the manual scene sorts
  (`scene_sorting/full_2015/`, `scene_sorting/subsample_2016/`), extracted at each
  recording's `scene1_time_s` from the CamHD archive.
- Monthly spread: 2015 = 2 per month (Jul-Dec); 2016 = all confirmed subsample Scene-1.

## Workflow (once the count target is defined)

1. Open `notebooks/29_clickcount_clear_window.ipynb`, set
   `BASE = .../validation/early_recovery_handcount_2015_2016`, kernel
   `joseph-scaleworm-thesis`.
2. Click each worm (per the agreed target); counts + click coords save to
   `handcount_sheet.csv` and `clicks/`.
3. Score the detector: `MODEL=99_runs/scaleworm_v5_all/weights/best.pt LABEL=v5_early
   ... gate_retrain.py`-style scoring against these counts.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    picks = select()
    from collections import Counter

    print(f"selected {len(picks)} frames")
    print("  by year :", dict(Counter(p["year"] for p in picks)))
    print("  by month:", dict(sorted(Counter(p["month"] for p in picks).items())))
    if args.dry_run:
        return

    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    (OUT / "clicks").mkdir(parents=True, exist_ok=True)
    (OUT / "clicks" / ".gitkeep").touch()

    rows = []
    for p in picks:
        dst = OUT / "frames" / f"{p['frame_id']}.png"
        ready = dst.exists() or (
            p["scene1_time_s"] and extract(p["video_path"], float(p["scene1_time_s"]), dst)
        )
        rows.append(
            {
                "frame_id": p["frame_id"],
                "datetime_utc": p["frame_id"].split("-")[1],
                "camera_unit": f"CAMHDA301-{p['year']}",
                "quarter": f"{p['month'][:4]}-Q{(int(p['month'][4:6]) - 1) // 3 + 1}",
                "sharpness_survey": "",
                "scene1_time_s": p["scene1_time_s"],
                "frame_ready": "Y" if ready else "no-frame",
                "worm_count": "",
                "counter": "",
                "date_counted": "",
                "notes": "count target TBD (advisor) — do not count yet",
                "video_path": p["video_path"],
            }
        )
    fields = list(rows[0].keys())
    with (OUT / "handcount_sheet.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT / "README.md").write_text(README)
    n_ready = sum(1 for r in rows if r["frame_ready"] == "Y")
    print(f"wrote {OUT}/handcount_sheet.csv ({len(rows)} rows, {n_ready} frames extracted)")
    print(f"wrote {OUT}/README.md + clicks/")


if __name__ == "__main__":
    main()
