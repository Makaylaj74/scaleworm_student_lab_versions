"""Tests for the period-comparison statistics (worms-vs-temperature analysis)."""

import numpy as np
import pandas as pd

from scripts.compare_periods_worms_vs_temp import (
    cluster_bootstrap_mean,
    period_stats,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["total_worms", "n_slots"])


def test_period_stats_frame_weighted_mean():
    # 20 worms / 4 frames + 10 worms / 1 frame = 30 worms / 5 frames = 6.0
    df = _df([(20, 4), (10, 1)])
    rng = np.random.default_rng(0)
    out = period_stats(df, "x", rng)
    assert out["n_mondays"] == 2
    assert out["n_frames"] == 5
    assert out["worms"] == 30
    assert out["mean_per_frame"] == 6.0  # frame-weighted, not the 10.5 Monday-mean


def test_ci_brackets_point_estimate_and_is_ordered():
    df = _df([(8, 2), (12, 3), (5, 1), (20, 4)])
    totals = df["total_worms"].to_numpy(float)
    slots = df["n_slots"].to_numpy(float)
    point = totals.sum() / slots.sum()
    lo, hi = cluster_bootstrap_mean(totals, slots, np.random.default_rng(20260916))
    assert lo <= point <= hi
    assert lo < hi


def test_bootstrap_is_seed_deterministic():
    df = _df([(8, 2), (12, 3), (5, 1), (20, 4)])
    totals = df["total_worms"].to_numpy(float)
    slots = df["n_slots"].to_numpy(float)
    a = cluster_bootstrap_mean(totals, slots, np.random.default_rng(20260916))
    b = cluster_bootstrap_mean(totals, slots, np.random.default_rng(20260916))
    assert a == b


def test_single_monday_ci_is_degenerate():
    # one Monday -> every resample is identical -> CI collapses to the point value
    df = _df([(15, 3)])
    totals = df["total_worms"].to_numpy(float)
    slots = df["n_slots"].to_numpy(float)
    lo, hi = cluster_bootstrap_mean(totals, slots, np.random.default_rng(1))
    assert lo == hi == 5.0
