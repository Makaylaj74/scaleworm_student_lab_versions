"""Tests for the Tuesday scale-worm series builder."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_tuesday_series as bts


def test_fit_ols_recovers_known_line():
    # y = 3 + 2x exactly -> perfect fit
    x = np.arange(10, dtype=float)
    y = 3 + 2 * x
    fit = bts.fit_ols(x, y)
    assert fit.b0 == pytest.approx(3.0)
    assert fit.b1 == pytest.approx(2.0)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.rse == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(fit.predict(x), y)


def test_fit_ols_needs_three_points():
    with pytest.raises(ValueError):
        bts.fit_ols(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny manifest + AI-count pair covering all frame_status branches."""
    manual = pd.DataFrame(
        [
            # Tuesday A: two counted slots (SEM defined) -> also paired with AI
            {
                "frame_id": "F1",
                "datetime_utc": "2022-01-04T00:15:00",
                "frame_status": "counted",
                "worm_count": 20,
            },
            {
                "frame_id": "F2",
                "datetime_utc": "2022-01-04T03:15:00",
                "frame_status": "counted",
                "worm_count": 30,
            },
            # Tuesday B: one counted slot (SEM undefined)
            {
                "frame_id": "F3",
                "datetime_utc": "2022-01-11T00:15:00",
                "frame_status": "counted",
                "worm_count": 10,
            },
            # a blur frame -> labeled, must be excluded from ai_only
            {
                "frame_id": "F4",
                "datetime_utc": "2022-01-11T03:15:00",
                "frame_status": "unusable_blur",
                "worm_count": "",
            },
        ]
    )
    ai = pd.DataFrame(
        [
            {"frame_id": "F1", "date": "2022-01-04", "ai_count": 8},
            {"frame_id": "F2", "date": "2022-01-04", "ai_count": 12},
            {"frame_id": "F3", "date": "2022-01-11", "ai_count": 4},
            {
                "frame_id": "F4",
                "date": "2022-01-11",
                "ai_count": 5,
            },  # labeled-blur, skip
            # AI-only frames on a NEW Tuesday C (never hand-labeled)
            {"frame_id": "F5", "date": "2022-01-18", "ai_count": 6},
            {"frame_id": "F6", "date": "2022-01-18", "ai_count": 9},
        ]
    )
    mp = tmp_path / "manual.csv"
    ap = tmp_path / "ai.csv"
    manual.to_csv(mp, index=False)
    ai.to_csv(ap, index=False)
    return mp, ap


def test_load_pairs_only_counted_with_ai(tmp_path):
    mp, ap = _write_fixture(tmp_path)
    pairs = bts.load_pairs(mp, ap)
    # F1,F2,F3 are counted & have AI; F4 is blur; F5,F6 have no manual
    assert set(pairs["frame_id"]) == {"F1", "F2", "F3"}
    assert pairs.loc[pairs.frame_id == "F1", "worm_count"].item() == 20


def test_load_ai_only_excludes_all_labeled(tmp_path):
    mp, ap = _write_fixture(tmp_path)
    extra = bts.load_ai_only(mp, ap)
    # only F5, F6 (F4 is blur-labeled and must be excluded even though AI-scored)
    assert set(extra["frame_id"]) == {"F5", "F6"}


def test_aggregate_manual_sem_and_ci(tmp_path):
    mp, _ap = _write_fixture(tmp_path)
    man = bts.aggregate_manual(mp)
    a = man[man.date == "2022-01-04"].iloc[0]
    assert a["n_slots"] == 2
    assert a["mean_worms"] == pytest.approx(25.0)
    assert a["total_worms"] == 50
    # n=2 bootstrap CI is bounded to the data support (never negative, never explodes)
    assert 20.0 <= a["ci_lo"] <= a["ci_hi"] <= 30.0
    # single-slot Tuesday: SEM/CI blank, not zero
    b = man[man.date == "2022-01-11"].iloc[0]
    assert b["n_slots"] == 1
    assert b["sem_worms"] == ""
    assert b["ci_lo"] == ""


def test_bootstrap_corrected_is_seeded_and_bracketed(tmp_path):
    mp, ap = _write_fixture(tmp_path)
    pairs = bts.load_pairs(mp, ap)
    ai_only = bts.load_ai_only(mp, ap)
    fit = bts.fit_ols(pairs["ai_count"].to_numpy(), pairs["worm_count"].to_numpy())
    c1 = bts.bootstrap_corrected(pairs, ai_only, fit, n_boot=500, seed=123)
    c2 = bts.bootstrap_corrected(pairs, ai_only, fit, n_boot=500, seed=123)
    # deterministic under a fixed seed
    pd.testing.assert_frame_equal(c1, c2)
    row = c1[c1.date == "2022-01-18"].iloc[0]
    assert row["n_slots"] == 2
    assert row["source"] == "ai_corrected"
    # point estimate lies inside its own 95% CI, and the CI is ordered
    assert row["ci_lo"] <= row["mean_worms"] <= row["ci_hi"]
    assert row["ci_lo"] < row["ci_hi"]


def test_corrected_counts_never_negative(tmp_path):
    mp, ap = _write_fixture(tmp_path)
    pairs = bts.load_pairs(mp, ap)
    ai_only = bts.load_ai_only(mp, ap)
    fit = bts.fit_ols(pairs["ai_count"].to_numpy(), pairs["worm_count"].to_numpy())
    c = bts.bootstrap_corrected(pairs, ai_only, fit, n_boot=200, seed=7)
    assert (c["mean_worms"] >= 0).all()
    assert (c["ci_lo"] >= 0).all()
