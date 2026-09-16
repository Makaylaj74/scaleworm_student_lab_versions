"""Tests for the period-comparison statistics (worms-vs-temperature analysis)."""

import numpy as np
import pandas as pd

from scripts import compare_periods_worms_vs_temp as cp
from scripts.compare_periods_worms_vs_temp import (
    cluster_bootstrap_mean,
    load_ai2019,
    period_stats,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["total_worms", "n_slots"])


def test_load_ai2019_empty_when_absent(tmp_path, monkeypatch):
    # missing AI-corrected file -> empty frame (overlay simply not drawn), no crash
    monkeypatch.setattr(cp, "AI2019", tmp_path / "does_not_exist.csv")
    out = load_ai2019()
    assert out.empty
    assert "mean_corrected" in out.columns


def test_load_ai2019_parses_and_sorts(tmp_path, monkeypatch):
    csv = tmp_path / "ai.csv"
    csv.write_text(
        "date,n_slots,mean_corrected,sem_corrected\n"
        "2019-02-04,5,26.8,3.1\n"
        "2019-01-07,5,37.1,4.4\n"
    )
    monkeypatch.setattr(cp, "AI2019", csv)
    out = load_ai2019()
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2019-01-07", "2019-02-04"]
    assert out["sem"].tolist() == [4.4, 3.1]


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
