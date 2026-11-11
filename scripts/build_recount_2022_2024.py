"""Blind stratified recount packet for the 2022-2024 Monday series (annotator-drift check).

The Sep-Dec 2021 blind recount (build_recount_sepdec_2021.py, 2026-11-11) found the
September-2026 box corrections were anchored on the v2 pre-labels: recount/original =
2.71 [2.35, 3.15], 15/15 frames higher. This measures whether the same drift holds across
the rest of the series, PER PERIOD, before deciding whether to recount all 714 frames.

Strata (one frame per randomly drawn Monday, seeded; n per stratum in STRATA):
  2022 H1, 2022 H2 (dense peak), 2023 clear (v2), 2023 blurry (v3, >= 2023-08-10), 2024 (v3).

Blind: copies the image and a FRESH pre-label from the SAME model the frame originally got
(v2 clear / v3 blurry, mirroring prelabel_manual_series.py), blanks every annotation column,
and skips any frame already shown to raters in validation/irr_round_*.

    .venv/bin/python3 scripts/build_recount_2022_2024.py
"""

from __future__ import annotations

import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "validation/monday_manual_series"
OUT = REPO / "validation/recount_2022_2024"
MODELS = {
    "v2": REPO / "99_runs/scaleworm_v2/weights/best.pt",
    "v3": REPO / "99_runs/scaleworm_v3/weights/best.pt",
}
CONF = 0.25
SEED = 20221111
BLUR_ONSET = "20230810"
BLANK = (
    "frame_status",
    "worm_count",
    "label_path",
    "annotation_method",
    "prelabel_model",
    "counter",
    "date_counted",
    "notes",
)
# stratum -> (first YYYYMMDD inclusive, last YYYYMMDD exclusive, n Mondays)
STRATA = {
    "2022_h1": ("20220101", "20220701", 4),
    "2022_h2_peak": ("20220701", "20230101", 5),
    "2023_clear": ("20230101", BLUR_ONSET, 5),
    "2023_blurry": (BLUR_ONSET, "20240101", 4),
    "2024": ("20240101", "20250101", 6),
}


def stratum_of(day: str) -> str | None:
    """Return the stratum a YYYYMMDD day falls in, or None."""
    for name, (lo, hi, _) in STRATA.items():
        if lo <= day < hi:
            return name
    return None


def irr_frame_ids() -> set[str]:
    """Every CAMHD frame id mentioned in any IRR round's CSVs (already seen by raters)."""
    pat = re.compile(r"CAMHDA301-\d{8}T\d{6}")
    ids: set[str] = set()
    for p in (REPO / "validation").glob("irr_round_*/**/*.csv"):
        ids.update(pat.findall(p.read_text(errors="ignore")))
    return ids


def select(
    rows: list[dict], exclude: set[str], seed: int = SEED
) -> list[tuple[str, dict]]:
    """Draw n Mondays per stratum, then one counted frame per drawn Monday."""
    days: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["frame_status"] != "counted" or r["frame_id"] in exclude:
            continue
        day = r["frame_id"][10:18]
        s = stratum_of(day)
        if s:
            days[s][day].append(r)
    rng = random.Random(seed)
    picks = []
    for name, (_, _, n) in STRATA.items():
        for day in sorted(rng.sample(sorted(days[name]), n)):
            picks.append((name, rng.choice(days[name][day])))
    return picks


def main() -> None:
    from ultralytics import YOLO

    with (SRC / "frame_manifest.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    picks = select(rows, irr_frame_ids())

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    models = {k: YOLO(str(p)) for k, p in MODELS.items()}
    out_rows = []
    for name, r in picks:
        fid, mkey = r["frame_id"], r["prelabel_model"]
        png = OUT / "images" / f"{fid}.png"
        shutil.copy2(SRC / "images" / f"{fid}.png", png)
        res = models[mkey](str(png), conf=CONF, verbose=False)[0]
        lines = [
            f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for x, y, w, h in res.boxes.xywhn.tolist()
        ]
        (OUT / "labels" / f"{fid}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        row = {**r, **{c: "" for c in BLANK}}
        row.update(
            frame_status="prelabeled",
            label_path=f"labels/{fid}.txt",
            prelabel_model=mkey,
            notes=f"blind recount 2026-11 (drift check); stratum={name}",
        )
        out_rows.append(row)

    with (OUT / "frame_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    counts = defaultdict(int)
    for name, _ in picks:
        counts[name] += 1
    print(f"wrote {len(out_rows)} frames -> {OUT}: {dict(counts)}")


if __name__ == "__main__":
    main()
