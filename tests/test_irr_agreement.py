"""Unit tests for ``scripts/irr_agreement.py``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import irr_agreement as irr
import numpy as np
import pytest
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss


def _write_log(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """Write a sort_log.csv from (stem, decision, scene1_time_s) tuples."""
    lines = ["stem,decision,scene1_time_s,decided_at_utc"]
    for stem, decision, t in rows:
        lines.append(f"{stem},{decision},{t},2026-07-27T00:00:00+00:00")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_decisions_last_row_wins_and_parses_time(tmp_path):
    log = _write_log(
        tmp_path / "sort_log.csv",
        [
            ("A", "skip", ""),
            ("A", "scene1", "210"),  # later row overrides the skip
            ("B", "not_scene1", ""),
        ],
    )
    decisions = irr.load_decisions(log)
    assert decisions["A"] == {"decision": "scene1", "time": 210.0}
    assert decisions["B"] == {"decision": "not_scene1", "time": None}


def test_load_decisions_missing_file_is_empty(tmp_path):
    assert irr.load_decisions(tmp_path / "nope.csv") == {}


def test_cohen_kappa_perfect_and_chance():
    assert irr.cohen_kappa(["scene1", "not_scene1"], ["scene1", "not_scene1"]) == 1.0
    # totally independent 50/50 pattern -> kappa near 0
    a = ["scene1", "scene1", "not_scene1", "not_scene1"]
    b = ["scene1", "not_scene1", "scene1", "not_scene1"]
    assert irr.cohen_kappa(a, b) == pytest.approx(0.0, abs=1e-9)


def test_cohen_kappa_empty_is_nan():
    assert np.isnan(irr.cohen_kappa([], []))


def test_fleiss_kappa_matches_statsmodels():
    rng = np.random.default_rng(0)
    # 20 items, 3 raters, 2 categories -> random count table with row sums == 3
    table = np.zeros((20, 2))
    for i in range(20):
        k = rng.integers(0, 4)  # scene1 votes among 3 raters
        table[i] = [k, 3 - k]
    assert irr.fleiss_kappa(table) == pytest.approx(sm_fleiss(table), abs=1e-9)


def test_fleiss_kappa_unequal_rows_is_nan():
    assert np.isnan(irr.fleiss_kappa(np.array([[3.0, 0.0], [2.0, 0.0]])))


def test_bootstrap_ci_deterministic_and_ordered():
    items = list(range(100))
    ci_a = irr.bootstrap_ci(items, lambda s: float(np.mean(s)), n_boot=200, seed=42)
    ci_b = irr.bootstrap_ci(items, lambda s: float(np.mean(s)), n_boot=200, seed=42)
    assert ci_a == ci_b
    assert ci_a[0] <= ci_a[1]


def test_bootstrap_ci_too_few_items_is_nan():
    lo, hi = irr.bootstrap_ci([1.0], lambda s: float(np.mean(s)))
    assert np.isnan(lo) and np.isnan(hi)


def _three_perfect_raters() -> dict:
    """Three raters agreeing perfectly on a mixed binary pattern, times within tol."""
    logs = {r: {} for r in ("rater_a", "rater_b", "rater_c")}
    for stem, dec in [("A", "scene1"), ("B", "not_scene1"), ("C", "scene1")]:
        for log in logs.values():
            log[stem] = {
                "decision": dec,
                "time": 180.0 if dec == "scene1" else None,
            }
    return logs


def test_compute_report_perfect_agreement():
    logs = _three_perfect_raters()
    rep = irr.compute_report(logs, tolerance_s=30.0, n_boot=100, seed=1)
    assert rep["binary"]["n_included"] == 3
    assert rep["binary"]["observed_agreement"] == 1.0
    # perfect but mixed categories -> kappa == 1.0
    assert rep["binary"]["fleiss_kappa"] == pytest.approx(1.0)
    for d in rep["binary"]["pairwise_cohen"].values():
        assert d["cohen_kappa"] == pytest.approx(1.0)
    # two scene1 stems, identical times -> within tolerance, zero diff
    assert rep["time"]["n_included"] == 2
    assert rep["time"]["within_tolerance_rate"] == 1.0
    assert rep["time"]["mean_abs_pairwise_diff_s"] == pytest.approx(0.0)


def test_compute_report_excludes_skips_and_incomplete():
    logs = {
        "rater_a": {"A": {"decision": "scene1", "time": 100.0}},
        "rater_b": {"A": {"decision": "skip", "time": None}},  # skip -> excluded
        "rater_c": {"A": {"decision": "scene1", "time": 100.0}},
    }
    rep = irr.compute_report(logs, n_boot=50, seed=1)
    assert rep["binary"]["n_included"] == 0  # A dropped because rater_b skipped
    assert rep["time"]["n_included"] == 0


def test_compute_report_time_tolerance_boundary():
    # all scene1; times span exactly the tolerance and just over it
    logs = {
        "rater_a": {
            "A": {"decision": "scene1", "time": 100.0},
            "B": {"decision": "scene1", "time": 100.0},
        },
        "rater_b": {
            "A": {"decision": "scene1", "time": 130.0},
            "B": {"decision": "scene1", "time": 131.0},
        },
        "rater_c": {
            "A": {"decision": "scene1", "time": 115.0},
            "B": {"decision": "scene1", "time": 100.0},
        },
    }
    rep = irr.compute_report(logs, tolerance_s=30.0, n_boot=50, seed=1)
    assert rep["time"]["n_included"] == 2
    # A spans 30 (<=30, agree); B spans 31 (>30, disagree) -> 0.5
    assert rep["time"]["within_tolerance_rate"] == pytest.approx(0.5)
