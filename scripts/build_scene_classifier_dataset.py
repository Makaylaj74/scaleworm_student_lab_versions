"""Assemble a per-frame Scene-1 classifier dataset from the hand-sort logs.

Turns the accumulated `scene_sorting/**/sort_log.csv` decisions (recording-level
Scene-1/not + `scene1_time_s`) into a labeled per-tile image manifest for a binary
"is this frame the front-on Scene-1 view?" classifier. Running that classifier on
all ~30 tiles of a new recording yields both outputs the pipeline needs: whether
Scene-1 appears, and at what clip time (the top-scoring tile).

Labeling rule (deliberately conservative — the sort labels are recording-level):
  - decision == scene1, time T  -> ONE positive tile at t=T (the annotator's pick;
    all Scene-1 times are exact 30 s tile multiples).
  - decision == not_scene1      -> NEG_PER_RECORDING negatives sampled from the
    standard 30 s tile grid (30..840 s; t=0 is startup/black and skipped).
  - other tiles of a scene1 recording -> UNLABELED (the view may linger near T;
    labeling them risks noise), so they are not emitted.
  - decision == skip            -> dropped.

Dedup: recordings sorted in more than one log (the validation re-label batches) are
kept once, preferring the primary (non-validation) batch. The 7 recordings whose
passes DISAGREE are dropped as ambiguous.

Split: DAY-level (all 8 slots of a date share a split) to prevent same-day
near-duplicate leakage -- the same rule the detector-recall work uses. Deterministic
hash on the date, 70/15/15 train/val/test.

Output: datasets/scene_classifier/labels.csv  (+ dataset_summary.json)
No images are extracted here -- see extract_scene_classifier_frames.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARCHIVE_BASE = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
OUT_DIR = REPO / "datasets/scene_classifier"

# validation re-label batches are secondary to their primary full-day sorts
PRIMARY_RANK = {
    "full_2023_2024": 0,
    "tuesdays_2021_2024": 0,
    "clear_window_2021_2022": 0,
    "2019_2020": 0,
    "validation_2023_03": 1,
    "validation_tuesdays": 1,
}
BLUR_ONSET = date(2023, 8, 10)
NEG_TILE_GRID = tuple(range(30, 841, 30))  # skip t=0 (startup/black)
NEG_PER_RECORDING = 4
SEED = 20260922


def stem_to_dt(stem: str) -> datetime:
    """`CAMHDA301-20230808T061500` -> datetime."""
    return datetime.strptime(stem.split("-", 1)[1], "%Y%m%dT%H%M%S")  # noqa: DTZ007 (naive UTC stem; used only for date math)


def stem_to_video(stem: str, base: Path = ARCHIVE_BASE) -> Path:
    dt = stem_to_dt(stem)
    return base / f"{dt:%Y/%m/%d}" / f"{stem}.mp4"


def load_decisions(sort_logs_glob: str) -> dict[str, dict]:
    """Load + dedup sort-log decisions keyed by recording stem.

    Prefers the primary batch for re-labeled recordings; drops recordings whose
    passes disagree on scene1-vs-not.
    """
    by_stem: dict[str, list[dict]] = {}
    for p in sorted(glob.glob(sort_logs_glob, recursive=True)):
        batch = Path(p).parent.name
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["decision"] == "skip":
                    continue
                by_stem.setdefault(r["stem"], []).append({"batch": batch, **r})

    out: dict[str, dict] = {}
    dropped_ambiguous = 0
    for stem, recs in by_stem.items():
        decisions = {r["decision"] for r in recs}
        if len(decisions) > 1:
            dropped_ambiguous += 1  # passes disagree -> ambiguous, drop
            continue
        best = min(recs, key=lambda r: PRIMARY_RANK.get(r["batch"], 9))
        out[stem] = best
    out["_dropped_ambiguous"] = dropped_ambiguous  # sentinel; popped by caller
    return out


def day_split(day: str, seed: int = SEED) -> str:
    h = hashlib.md5(f"{seed}:{day}".encode()).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return "train" if v < 0.70 else ("val" if v < 0.85 else "test")


def build_rows(
    decisions: dict[str, dict],
    neg_per: int = NEG_PER_RECORDING,
    seed: int = SEED,
    base: Path = ARCHIVE_BASE,
) -> list[dict]:
    import random

    rng = random.Random(seed)
    rows = []
    for stem, r in sorted(decisions.items()):
        dt = stem_to_dt(stem)
        day = dt.date().isoformat()
        era = "clear" if dt.date() < BLUR_ONSET else "blurry"
        split = day_split(day, seed)
        video = str(stem_to_video(stem, base))
        common = {
            "stem": stem,
            "date": day,
            "video_path": video,
            "era": era,
            "split": split,
            "source_batch": r["batch"],
        }
        if r["decision"] == "scene1" and r["scene1_time_s"]:
            rows.append(
                {**common, "tile_time_s": int(float(r["scene1_time_s"])), "label": 1}
            )
        elif r["decision"] == "not_scene1":
            for t in rng.sample(NEG_TILE_GRID, min(neg_per, len(NEG_TILE_GRID))):
                rows.append({**common, "tile_time_s": t, "label": 0})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sort-logs", default=str(REPO / "scene_sorting/**/sort_log.csv"))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--neg-per", type=int, default=NEG_PER_RECORDING)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--base", type=Path, default=ARCHIVE_BASE)
    args = ap.parse_args()

    decisions = load_decisions(args.sort_logs)
    dropped = decisions.pop("_dropped_ambiguous")
    rows = build_rows(decisions, args.neg_per, args.seed, args.base)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = args.out_dir / "labels.csv"
    fields = [
        "stem",
        "date",
        "video_path",
        "tile_time_s",
        "label",
        "era",
        "split",
        "source_batch",
    ]
    with labels_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def tally(key):
        from collections import Counter

        return dict(Counter(r[key] for r in rows))

    # verify no date leaks across splits
    day_to_splits: dict[str, set] = {}
    for r in rows:
        day_to_splits.setdefault(r["date"], set()).add(r["split"])
    leaks = [d for d, s in day_to_splits.items() if len(s) > 1]

    summary = {
        "recordings_used": len(decisions),
        "recordings_dropped_ambiguous": dropped,
        "total_labeled_tiles": len(rows),
        "positives": sum(r["label"] for r in rows),
        "negatives": sum(1 - r["label"] for r in rows),
        "by_split": tally("split"),
        "by_era": tally("era"),
        "pos_by_split": {
            s: sum(r["label"] for r in rows if r["split"] == s)
            for s in ("train", "val", "test")
        },
        "neg_per_recording": args.neg_per,
        "day_level_split_leaks": len(leaks),
        "seed": args.seed,
    }
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {labels_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
