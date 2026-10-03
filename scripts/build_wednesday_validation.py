"""Build a stratified Wednesday validation session for the Scene-1 classifier gate.

The 1363 rendered Wednesday contact sheets (`scene_sorting/wednesdays_2021_2024/`) are
unsorted and were never needed for counting (the scene-sort-free M-F harvest already
covers Wednesday). This draws a stratified ~N-recording SAMPLE into a nb25-sortable
session so MJ can sort it BLIND, then `gate_scene_classifier.py` scores the trained
Scene-1 classifier against her sort — an independent check that the Monday-validated
recall transfers to Wednesday (No Borrowed Assumptions).

Design is deliberately blind: the classifier's predictions are computed/locked in
advance (predictions.csv), MJ sorts without seeing them, and only then are the two
compared. The Wednesday manifest lacks `video_path`, so we add it (derived from the
stem) because the classifier reads the actual video to score tiles.

Usage:
    python scripts/build_wednesday_validation.py            # 3 per year x slot (~96)
    python scripts/build_wednesday_validation.py --per 2    # smaller
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "scene_sorting" / "wednesdays_2021_2024"
OUT = REPO / "scene_sorting" / "wednesday_validation"
ARCHIVE = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
SEED = 20261003
PER_STRATUM = 3


def video_path(stem: str) -> Path:
    """Archive path for a recording stem CAMHDA301-YYYYMMDDTHHMMSS."""
    d = stem.split("-")[1]  # YYYYMMDDTHHMMSS
    return ARCHIVE / d[:4] / d[4:6] / d[6:8] / f"{stem}.mp4"


def stratified_draw(
    rows: list[dict], per_stratum: int = PER_STRATUM, seed: int = SEED
) -> list[dict]:
    """Draw up to `per_stratum` recordings from each (year, slot) stratum, seeded.

    Deterministic: strata are processed in sorted key order and each is sampled with a
    freshly-seeded RNG, so the result depends only on (rows, per_stratum, seed).
    """
    strata: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        year = r["stem"].split("-")[1][:4]
        strata.setdefault((year, r["slot"]), []).append(r)
    picked: list[dict] = []
    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda r: r["stem"])
        rng = random.Random(f"{seed}-{key[0]}-{key[1]}")
        k = min(per_stratum, len(bucket))
        picked.extend(rng.sample(bucket, k))
    return sorted(picked, key=lambda r: r["stem"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per", type=int, default=PER_STRATUM, help="recordings per year x slot")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    rows = [
        r
        for r in csv.DictReader((args.src / "manifest.csv").open())
        if r["video_exists"] == "True"
    ]
    picked = stratified_draw(rows, args.per, args.seed)

    # fresh session dirs (nb25 layout)
    for sub in ("contact_sheets", "scene1", "not_scene1", "unusable_blur"):
        d = args.out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    missing_sheet, missing_video = [], []
    out_rows = []
    for r in picked:
        stem = r["stem"]
        sheet = args.src / "contact_sheets" / f"{stem}.png"
        if sheet.exists():
            shutil.copy2(sheet, args.out / "contact_sheets" / f"{stem}.png")
        else:
            missing_sheet.append(stem)
            continue
        vp = video_path(stem)
        if not vp.exists():
            missing_video.append(stem)
        out_rows.append({**r, "video_path": str(vp)})

    fields = list(rows[0].keys()) + ["video_path"]
    with (args.out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    years = sorted({r["stem"].split("-")[1][:4] for r in out_rows})
    print(f"Wednesday validation session: {len(out_rows)} recordings "
          f"({args.per}/stratum, {len(years)} years {years}) -> {args.out}")
    if missing_sheet:
        print(f"  WARNING missing contact sheets ({len(missing_sheet)}): {missing_sheet}")
    if missing_video:
        print(f"  WARNING missing videos ({len(missing_video)}): {missing_video}")
    print("NEXT: 1) pre-run the classifier (locks predictions in BEFORE sorting):")
    print(f"        /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 "
          f"scripts/gate_scene_classifier.py --session {args.out}")
    print("      2) MJ sorts BLIND in nb25 (SESSION_DIR = wednesday_validation)")
    print("      3) re-run the same gate command -> gate_results.md (kappa, agreement, timing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
