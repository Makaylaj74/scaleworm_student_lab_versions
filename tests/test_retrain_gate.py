"""Tests for the detector-retrain leakage + scoring helpers.

Covers the pure logic that the retrain's correctness depends on: greedy point-in-box
matching, the day/era parsing used for leakage exclusion, YOLO label validation, and the
counted-stems filter.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import build_retrain_dataset as brd
import gate_retrain as gr


# --- day/era parsing (drives day-level leakage exclusion) ---
def test_day_of():
    assert gr.day_of("CAMHDA301-20230116T031500_01") == "20230116"
    assert brd.day_of("CAMHDA301-20211015T211500") == "20211015"


def test_era_split_on_swap_day():
    # SWAP_DAY = 20230811: strictly-before is clear, on/after is blurry
    assert brd.era_of("CAMHDA301-20230810T001500") == "clear"
    assert brd.era_of("CAMHDA301-20230811T001500") == "blurry"
    assert brd.era_of("CAMHDA301-20240205T001500") == "blurry"


# --- greedy point-in-box matching ---
def test_point_match_basic():
    # one box around (5,5), one click inside -> 1 tp, 0 fp, 0 fn
    preds = [(0, 0, 10, 10, 0.9)]
    pts = [(5, 5)]
    assert gr.point_match(preds, pts) == (1, 0, 0)


def test_point_match_fp_and_fn():
    # box with no click inside = fp; click outside all boxes = fn
    preds = [(0, 0, 10, 10, 0.9)]
    pts = [(50, 50)]
    tp, fp, fn = gr.point_match(preds, pts)
    assert (tp, fp, fn) == (0, 1, 1)


def test_point_match_each_click_claimed_once():
    # two overlapping boxes, one click -> only one box matches, other is fp
    preds = [(0, 0, 10, 10, 0.9), (0, 0, 10, 10, 0.5)]
    pts = [(5, 5)]
    assert gr.point_match(preds, pts) == (1, 1, 0)


def test_point_match_conf_order_deterministic():
    # highest-conf box claims the shared click first
    high = (0, 0, 4, 4, 0.9)
    low = (0, 0, 10, 10, 0.2)
    tp, fp, fn = gr.point_match([low, high], [(1, 1)])
    assert (tp, fp, fn) == (1, 1, 0)


# --- YOLO label validation ---
def test_validate_label_ok(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("0 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.05 0.05\n")
    assert brd.validate_label(p) == 2


def test_validate_label_empty_is_valid_negative(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("")
    assert brd.validate_label(p) == 0


def test_validate_label_rejects_bad_class(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("1 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError):
        brd.validate_label(p)


def test_validate_label_rejects_out_of_range(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("0 1.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError):
        brd.validate_label(p)


# --- counted-stems filter (accepts frame_id or stem, worm_count or human_count) ---
def test_counted_stems_filters_blank(tmp_path):
    p = tmp_path / "sheet.csv"
    p.write_text(
        "frame_id,worm_count\n"
        "CAMHDA301-20230116T031500,12\n"
        "CAMHDA301-20230117T031500,\n"
        "CAMHDA301-20230118T031500,nan\n"
    )
    assert brd.counted_stems(p) == {"CAMHDA301-20230116T031500"}


def test_boot_ci_empty_is_nan():
    import math

    lo, hi = gr.boot_ci([])
    assert math.isnan(lo) and math.isnan(hi)
