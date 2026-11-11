"""Score the era-fixed Scene-1-time finder (numbers cited in method_writeup_detector §5).

Reads the per-recording density reads in validation/era_time_finder/ (from the other
era_finder_*.py scripts) and the hand counts / scene sorts, then reports, with bootstrap
CIs (n = 10 000, seed 0, percentile):
  1. Monday 2021 (91 rule-counted frames): r and count ratio for the human-selected
     frame, the current harvest frame, and the era-time read, per period.
  2. Off-scene cutoff chosen on Monday Jan-Aug 2021 labels only (max Youden J).
  3. Tuesday (independent; rules fixed from Mondays): era-time agreement with the
     Tuesday sort, cutoff sensitivity/specificity, per-frame and per-day r vs Tuesday
     box counts for era-time vs harvest, and the paired difference in r.

    .venv/bin/python3 scripts/era_finder_score.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
D = REPO / "validation/era_time_finder"
PERFRAME = (
    REPO
    / "notebooks/worm_series_2021_2023_density_run/worm_series_2021_2023_density_per_frame.csv"
)
CLEAR_END = "20230810"
B, SEED = 10_000, 0


def boot_r_ratio(a: np.ndarray, h: np.ndarray, rng) -> str:
    """Pearson r and sum-ratio a/h with percentile bootstrap CIs over frames."""
    idx = rng.integers(0, len(a), (B, len(a)))
    rs = np.percentile([np.corrcoef(a[i], h[i])[0, 1] for i in idx], [2.5, 97.5])
    rt = np.percentile(a[idx].sum(1) / h[idx].sum(1), [2.5, 97.5])
    return (
        f"n={len(a)} r={np.corrcoef(a, h)[0, 1]:.2f} [{rs[0]:.2f}, {rs[1]:.2f}]  "
        f"ratio={a.sum() / h.sum():.2f} [{rt[0]:.2f}, {rt[1]:.2f}]"
    )


def boot_delta_r(a: np.ndarray, b: np.ndarray, h: np.ndarray, rng) -> str:
    """Paired bootstrap of r(a, h) - r(b, h)."""
    idx = rng.integers(0, len(h), (B, len(h)))
    d = [np.corrcoef(a[i], h[i])[0, 1] - np.corrcoef(b[i], h[i])[0, 1] for i in idx]
    lo, hi = np.percentile(d, [2.5, 97.5])
    obs = np.corrcoef(a, h)[0, 1] - np.corrcoef(b, h)[0, 1]
    return f"delta r={obs:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def youden_cutoff(pos: pd.Series, neg: pd.Series) -> float:
    grid = np.arange(0.5, 15, 0.5)
    return float(max(grid, key=lambda c: (pos >= c).mean() - (neg >= c).mean()))


def main() -> None:
    rng = np.random.default_rng(SEED)
    # 1. Mondays
    mon = pd.read_csv(D / "diag_janaug_density.csv")
    grid = pd.read_csv(D / "pilot_era_time.csv")
    at_t0 = grid[grid.t == grid.t0].set_index("frame_id").dh
    mon["era"] = mon.frame_id.map(at_t0)
    mon["period"] = np.where(mon.frame_id.str[10:16] < "202109", "Jan-Aug", "Sep-Dec")
    print("1. Monday 2021, rule hand counts")
    for col in ("dh_on_mj_frame", "dh_on_harvest_frame", "era"):
        for p, x in [*mon.groupby("period"), ("all", mon)]:
            print(
                f"   {col:20s} {p:8s}",
                boot_r_ratio(x[col].to_numpy(), x.worm_count.to_numpy(float), rng),
            )

    # 2. Cutoff from Monday labels only
    sort_mon = pd.read_csv(REPO / "scene_sorting/monday_janaug_2021_3slot/sort_log.csv")
    pos = at_t0[at_t0.index.isin(mon.frame_id)]
    neg = at_t0[at_t0.index.isin(sort_mon[sort_mon.decision == "not_scene1"].stem)]
    cut = youden_cutoff(pos, neg)
    print(
        f"2. cutoff={cut} (Monday: sens {(pos >= cut).mean():.2f}, "
        f"spec {(neg < cut).mean():.2f}, n={len(pos)}/{len(neg)})"
    )

    # 3. Tuesdays
    tue = pd.read_csv(D / "tuesday_check.csv")
    rd = tue.groupby("frame_id").dh.max()
    era_t = tue.groupby("frame_id").t.apply(set)
    s = pd.read_csv(REPO / "scene_sorting/tuesdays_2021_2024/sort_log.csv").set_index(
        "stem"
    )
    s = s[s.index.str[10:18] < CLEAR_END]
    s1 = s[s.decision == "scene1"]
    match = np.mean([t in era_t[f] for f, t in s1.scene1_time_s.items()])
    y = s.decision == "scene1"
    print(
        f"3. Tuesday: era time matches sort {match:.3f} (n={len(s1)}); cutoff sens "
        f"{(rd[s.index][y] >= cut).mean():.2f} spec {(rd[s.index][~y] < cut).mean():.2f} "
        f"(n={len(s)})"
    )
    box = pd.read_csv(
        REPO / "validation/tuesday_manual_series/box_correct_manifest.csv"
    )
    box = box[
        (box.frame_status == "counted") & (box.frame_id.str[10:18] < CLEAR_END)
    ].set_index("frame_id")
    pf = pd.read_csv(PERFRAME).set_index("frame_id")
    x = pd.DataFrame(
        {"era": rd, "harv": pf.worm_index, "hand": box.worm_count.astype(float)}
    ).dropna()
    print(f"   per frame (hand mean {x.hand.mean():.1f}/fr)")
    for col in ("era", "harv"):
        print(f"     {col:5s}", boot_r_ratio(x[col].to_numpy(), x.hand.to_numpy(), rng))
    print(
        "    ",
        boot_delta_r(x.era.to_numpy(), x.harv.to_numpy(), x.hand.to_numpy(), rng),
    )

    day = lambda idx: idx.str[10:18]
    hand_d = x.hand.groupby(day(x.index)).mean()
    era_d = rd[s.index][rd[s.index] >= cut]
    era_d = era_d.groupby(day(era_d.index)).mean()
    harv_d = pf.worm_index.groupby(day(pf.index)).mean()
    j = pd.concat(
        [hand_d.rename("hand"), era_d.rename("era"), harv_d.rename("harv")], axis=1
    ).dropna()
    print("   per day (era gated at cutoff; harvest = mean of all slots)")
    for col in ("era", "harv"):
        print(f"     {col:5s}", boot_r_ratio(j[col].to_numpy(), j.hand.to_numpy(), rng))
    print(
        "    ",
        boot_delta_r(j.era.to_numpy(), j.harv.to_numpy(), j.hand.to_numpy(), rng),
    )


if __name__ == "__main__":
    main()
