"""Out-of-sample-validated crowding correction for v5_all clear counts.

v5_all compresses dense clear frames (count-ratio ~65% at >22 worms/frame; higher imgsz
does NOT fix it -- it's a crowding ceiling, see experiment_imgsz_density.py). For an
abundance INDEX this is correctable: the raw AI count still rises monotonically with the
true count, just sub-linearly. We fit true ~= a * ai_raw^p (power law; p>1 undoes the
saturation) and VALIDATE it out-of-sample.

Data: CLEAR, out-of-sample (not in v5 training) manually-counted Monday frames, AI counts
from notebooks/ai_counts_v5_per_frame.csv (conf 0.40 clear). DAY-level 60/40 train/test
split so no same-day leakage between fit and validation.

Reports, on the held-out TEST split, count-ratio by density bin BEFORE vs AFTER correction,
and per-frame MAE. Writes the fitted params to validation/v5_density_correction.json.
"""

from __future__ import annotations

import csv
import glob
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MONDAY = REPO / "validation/monday_manual_series"
AI = REPO / "notebooks/ai_counts_v5_per_frame.csv"
OUT = REPO / "validation/v5_density_correction.json"
BLUR_ONSET = pd.Timestamp("2023-08-11")
SEED = 20260930
BINS = [(0, 8), (8, 15), (15, 22), (22, 999)]


def training_stems() -> set[str]:
    s = set()
    for ds in ("scaleworm_retrain_all", "scaleworm_retrain_clear"):
        for p in glob.glob(str(REPO / f"datasets/{ds}/labels/train/*.txt")):
            s.add(Path(p).stem)
    return s


def day_of(fid: str) -> str:
    return fid.split("-")[1][:8]


def ratio_by_bin(pairs):
    out = {}
    for lo, hi in BINS:
        sub = [(m, h) for m, h in pairs if lo <= h < hi]
        if sub:
            out[f"{lo}-{hi}"] = (
                len(sub),
                sum(h for _, h in sub) / len(sub),
                sum(m for m, _ in sub) / sum(h for _, h in sub),
            )
    return out


def main() -> None:
    train_stems = training_stems()
    man = {
        r["frame_id"]: r
        for r in csv.DictReader((MONDAY / "frame_manifest.csv").open())
        if (r.get("worm_count") or "").strip() not in ("", "nan")
    }
    ai = pd.read_csv(AI).set_index("frame_id")

    rows = []  # (fid, ai_raw, manual)
    for fid, r in man.items():
        if fid in train_stems or fid not in ai.index:
            continue
        if pd.to_datetime(r["datetime_utc"]) >= BLUR_ONSET:
            continue  # clear only
        rows.append((fid, int(ai.at[fid, "ai_count"]), int(float(r["worm_count"]))))
    print(f"clear out-of-sample frames: {len(rows)}")

    # DAY-level 60/40 split
    days = sorted({day_of(f) for f, _, _ in rows})
    rng = random.Random(SEED)
    rng.shuffle(days)
    test_days = set(days[: round(len(days) * 0.4)])
    tr = [(f, a, m) for f, a, m in rows if day_of(f) not in test_days]
    te = [(f, a, m) for f, a, m in rows if day_of(f) in test_days]
    print(f"fit on {len(tr)} frames / {len(days) - len(test_days)} days; "
          f"test on {len(te)} frames / {len(test_days)} days")

    # power fit: manual = a * ai^p  ->  log manual = log a + p log ai  (ai>0, manual>0)
    fa = np.array([a for _, a, m in tr if a > 0 and m > 0], float)
    fm = np.array([m for _, a, m in tr if a > 0 and m > 0], float)
    p, loga = np.polyfit(np.log(fa), np.log(fm), 1)
    a = float(np.exp(loga))
    print(f"\nfit: manual = {a:.3f} * ai^{p:.3f}")

    def correct(ai_raw):
        return a * ai_raw**p if ai_raw > 0 else 0.0

    print("\n=== TEST split: count-ratio by density (BEFORE -> AFTER) ===")
    raw_pairs = [(m_ai, m) for _, m_ai, m in te]
    cor_pairs = [(correct(m_ai), m) for _, m_ai, m in te]
    rb, cb = ratio_by_bin(raw_pairs), ratio_by_bin(cor_pairs)
    for k in rb:
        n, mean, rr = rb[k]
        _, _, cr = cb[k]
        print(f"  density {k:<7} n={n:3d} manual/fr={mean:5.1f}  {rr:6.1%} -> {cr:6.1%}")
    raw_mae = np.mean([abs(m_ai - m) for m_ai, m in raw_pairs])
    cor_mae = np.mean([abs(c - m) for c, m in cor_pairs])
    raw_over = sum(m for m, _ in raw_pairs) / sum(h for _, h in raw_pairs)
    cor_over = sum(m for m, _ in cor_pairs) / sum(h for _, h in cor_pairs)
    print(f"\n  overall count-ratio : {raw_over:.1%} -> {cor_over:.1%}")
    print(f"  per-frame MAE       : {raw_mae:.2f} -> {cor_mae:.2f} worms")

    OUT.write_text(json.dumps(
        {"model": "scaleworm_v5_all", "conf_clear": 0.40, "form": "manual = a * ai_raw**p",
         "a": round(a, 4), "p": round(p, 4), "seed": SEED,
         "fit_n": len(fa), "test_n": len(te),
         "test_overall_ratio_before": round(raw_over, 4),
         "test_overall_ratio_after": round(cor_over, 4),
         "test_mae_before": round(float(raw_mae), 3),
         "test_mae_after": round(float(cor_mae), 3),
         "note": "clear-era crowding correction; blurry needs no correction (94% raw)"},
        indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
