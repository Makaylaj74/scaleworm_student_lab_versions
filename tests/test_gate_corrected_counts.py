"""Tests for the pure logic in build_gate_corrected_counts (no detector I/O)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from build_gate_corrected_counts import correct, recommend, unit_of, weekly


def _gate(rows: list[dict]) -> pd.DataFrame:
    cols = [
        "model",
        "recall",
        "recall_lo",
        "recall_hi",
        "precision",
        "f1",
        "count_bias",
        "count_mae",
    ]
    return pd.DataFrame([{c: r.get(c, 0.0) for c in cols} for r in rows])


def test_unit_of_august_swap():
    # Jan-Jul footage belongs to the PREVIOUS August's camera.
    assert unit_of(datetime(2019, 3, 1, tzinfo=UTC)) == "CAMHDA301-2018"
    assert unit_of(datetime(2019, 7, 31, tzinfo=UTC)) == "CAMHDA301-2018"
    # Aug onward belongs to that year's camera.
    assert unit_of(datetime(2019, 8, 1, tzinfo=UTC)) == "CAMHDA301-2019"
    assert unit_of(datetime(2020, 12, 31, tzinfo=UTC)) == "CAMHDA301-2020"


def test_recommend_picks_passing_model():
    # v2 clears 0.40 (best count_mae among passers); v3 also passes but worse MAE.
    g = _gate(
        [
            {
                "model": "v2",
                "recall": 0.447,
                "recall_lo": 0.40,
                "recall_hi": 0.49,
                "count_mae": 3.95,
            },
            {
                "model": "v3",
                "recall": 0.42,
                "recall_lo": 0.35,
                "recall_hi": 0.48,
                "count_mae": 7.71,
            },
        ]
    )
    rec = recommend(g, min_recall=0.40)
    assert rec is not None
    assert rec["model"] == "v2"
    assert rec["correction"] == pytest.approx(1.0 / 0.447, rel=1e-6)


def test_recommend_tie_breaks_on_count_mae():
    # Both pass recall; the one with lower count_mae wins even if recall is lower.
    g = _gate(
        [
            {
                "model": "v2",
                "recall": 0.60,
                "recall_lo": 0.5,
                "recall_hi": 0.7,
                "count_mae": 5.0,
            },
            {
                "model": "v3",
                "recall": 0.45,
                "recall_lo": 0.4,
                "recall_hi": 0.5,
                "count_mae": 2.0,
            },
        ]
    )
    assert recommend(g, min_recall=0.40)["model"] == "v3"


def test_recommend_fail_when_none_clear_min_recall():
    g = _gate(
        [
            {
                "model": "v2",
                "recall": 0.17,
                "recall_lo": 0.07,
                "recall_hi": 0.24,
                "count_mae": 9.2,
            },
            {
                "model": "v3",
                "recall": 0.27,
                "recall_lo": 0.20,
                "recall_hi": 0.34,
                "count_mae": 6.8,
            },
        ]
    )
    assert recommend(g, min_recall=0.40) is None


def test_correct_point_and_bracket():
    rec = {"recall": 0.5, "recall_lo": 0.4, "recall_hi": 0.6, "correction": 2.0}
    point, lo, hi = correct(10, rec)
    assert point == pytest.approx(20.0)  # 10 / 0.5
    assert lo == pytest.approx(10 / 0.6)  # higher recall -> smaller estimate
    assert hi == pytest.approx(10 / 0.4)
    assert lo < point < hi


def test_correct_zero_count_stays_zero():
    rec = {"recall": 0.45, "recall_lo": 0.4, "recall_hi": 0.5, "correction": 2.22}
    assert correct(0, rec) == (0.0, 0.0, 0.0)


def test_weekly_mean_and_sem():
    rows = [
        {"date": "2019-03-04", "corrected_count": 10.0},
        {"date": "2019-03-04", "corrected_count": 20.0},
        {"date": "2019-03-11", "corrected_count": 5.0},
    ]
    out = weekly(rows)
    assert [r["date"] for r in out] == ["2019-03-04", "2019-03-11"]
    assert out[0]["mean_corrected"] == pytest.approx(15.0)
    assert out[0]["n_slots"] == 2
    assert out[1]["sem_corrected"] == ""  # single-frame week -> undefined SEM
