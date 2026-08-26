"""Unit tests for scripts/count_frames.py (pure logic — no model or ffmpeg needed)."""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

from count_frames import count_worms, parse_stem_dt, stem_to_video


def test_stem_to_video_maps_to_year_month_day():
    root = Path("/data/CAMHDA301")
    got = stem_to_video("CAMHDA301-20240205T031500", root)
    assert got == root / "2024" / "02" / "05" / "CAMHDA301-20240205T031500.mp4"


def test_parse_stem_dt():
    assert parse_stem_dt("CAMHDA301-20240205T031500") == datetime(2024, 2, 5, 3, 15, 0)


def test_count_worms_single_class_counts_every_box():
    names = {0: "scale_worm"}
    # three detections, all class 0 -> all counted
    assert count_worms([0, 0, 0], names) == 3


def test_count_worms_single_class_empty():
    assert count_worms([], {0: "scale_worm"}) == 0


def test_count_worms_multi_class_filters_by_name():
    names = {0: "scale_worm", 1: "not_worm", 2: "shrimp"}
    # only the scale_worm class (id 0) should count; 'not_worm' must NOT match
    assert count_worms([0, 1, 2, 0, 1], names) == 2


def test_count_worms_multi_class_exact_match_case_insensitive():
    names = {0: "Scale_Worm", 1: "background"}
    assert count_worms([0, 0, 1], names) == 2


def test_count_worms_not_worm_class_is_excluded():
    # regression: substring matching would wrongly count 'not_worm' (contains 'worm')
    names = {0: "scale_worm", 1: "not_worm"}
    assert count_worms([1, 1, 1], names) == 0


@pytest.mark.parametrize("bad", ["CAMHDA301", "no-dash-parts", "CAMHDA301-2024"])
def test_parse_stem_dt_rejects_malformed(bad):
    with pytest.raises((ValueError, IndexError)):
        parse_stem_dt(bad)
