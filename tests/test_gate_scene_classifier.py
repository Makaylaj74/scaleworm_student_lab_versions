"""Unit tests for the gate's no-sklearn statistics (kappa, Wilson CI)."""

import math

import pytest

from scripts.gate_scene_classifier import cohen_kappa, wilson_ci


def test_kappa_perfect_agreement():
    a = ["scene1", "not_scene1", "scene1", "not_scene1"]
    assert cohen_kappa(a, a) == pytest.approx(1.0)


def test_kappa_chance_agreement_is_zero():
    # marginals equal, agreement == expected -> kappa 0
    a = ["scene1", "scene1", "not_scene1", "not_scene1"]
    b = ["scene1", "not_scene1", "scene1", "not_scene1"]
    assert cohen_kappa(a, b) == pytest.approx(0.0, abs=1e-9)


def test_kappa_total_disagreement_negative():
    a = ["scene1", "scene1", "not_scene1", "not_scene1"]
    b = ["not_scene1", "not_scene1", "scene1", "scene1"]
    assert cohen_kappa(a, b) < 0


def test_kappa_matches_known_value():
    # 8 agree-scene1, 4 agree-not, 2 human-scene1/model-not, 1 human-not/model-scene1
    a = ["scene1"] * 10 + ["not_scene1"] * 5
    b = ["scene1"] * 8 + ["not_scene1"] * 2 + ["not_scene1"] * 4 + ["scene1"] * 1
    # po = 12/15; compute pe from marginals and compare to direct formula
    n = 15
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = (a.count("scene1") / n) * (b.count("scene1") / n) + (
        a.count("not_scene1") / n
    ) * (b.count("not_scene1") / n)
    assert cohen_kappa(a, b) == pytest.approx((po - pe) / (1 - pe))


def test_wilson_ci_brackets_point_estimate():
    lo, hi = wilson_ci(18, 20)
    assert lo < 18 / 20 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_ci_full_agreement_upper_at_one_ish():
    lo, hi = wilson_ci(20, 20)
    assert hi == pytest.approx(1.0, abs=1e-9) or hi < 1.0
    assert lo < 1.0  # not degenerate


def test_wilson_ci_empty():
    lo, hi = wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)
