"""Tests for scripts/gate_v6_blurry.py (v6 blurry acceptance gate)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import gate_v6_blurry as g


def test_box_centers_denormalizes(tmp_path):
    lbl = tmp_path / "f.txt"
    # two boxes: centres at (0.25,0.50) and (0.75,0.10) in a 100x200 image
    lbl.write_text("0 0.25 0.50 0.1 0.1\n0 0.75 0.10 0.1 0.1\n")
    pts = g.box_centers(lbl, W=100, H=200)
    assert pts == [(25.0, 100.0), (75.0, 20.0)]


def test_box_centers_skips_blank_lines(tmp_path):
    lbl = tmp_path / "f.txt"
    lbl.write_text("0 0.5 0.5 0.1 0.1\n\n  \n")
    assert g.box_centers(lbl, W=10, H=10) == [(5.0, 5.0)]


def test_report_metrics():
    # 2 frames: gt=10 each; model finds 8 TP + 1 FP on one, 6 TP + 0 FP on other
    recs = [
        {"gt": 10, "pred": 9, "tp": 8, "fp": 1, "fn": 2},
        {"gt": 10, "pred": 6, "tp": 6, "fp": 0, "fn": 4},
    ]
    out = g.report("m", "BLURRY (Tuesday-dense, MJ-box GT)", recs)
    assert out["n"] == 2
    assert out["human"] == 20
    assert out["model_total"] == 15
    assert out["count_recall"] == 0.75  # 15/20
    assert out["point_recall"] == 0.70  # 14 TP / 20 gt
    assert round(out["precision"], 4) == round(14 / 15, 4)
    assert out["bias"] == -2.5  # mean((9-10),(6-10))


def test_models_and_conf_defined():
    assert set(g.MODELS) == {"v0_mushroom", "v5_all", "v6_blurry"}
    assert g.CONF["clear"] == 0.40 and g.CONF["blurry"] == 0.25
