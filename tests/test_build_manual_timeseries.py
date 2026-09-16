"""Tests for the Monday manual-series aggregation."""

import math
from datetime import date

from scripts.build_manual_timeseries import _parse_count, aggregate_by_monday


def test_groups_by_day_and_sorts():
    recs = [
        (date(2023, 1, 9), 5),
        (date(2023, 1, 2), 3),
        (date(2023, 1, 2), 7),
    ]
    out = aggregate_by_monday(recs)
    assert [r["date"] for r in out] == ["2023-01-02", "2023-01-09"]


def test_mean_total_and_slot_count():
    recs = [(date(2023, 1, 2), 4), (date(2023, 1, 2), 6), (date(2023, 1, 2), 8)]
    (row,) = aggregate_by_monday(recs)
    assert row["n_slots"] == 3
    assert row["total_worms"] == 18
    assert row["mean_worms"] == 6.0


def test_sem_is_sample_std_over_sqrt_n():
    # counts 2,4,6 -> mean 4, sample std 2.0, sem = 2/sqrt(3)
    recs = [(date(2023, 1, 2), 2), (date(2023, 1, 2), 4), (date(2023, 1, 2), 6)]
    (row,) = aggregate_by_monday(recs)
    assert row["std_worms"] == 2.0
    assert math.isclose(row["sem_worms"], 2.0 / math.sqrt(3), rel_tol=1e-6)


def test_single_slot_has_blank_spread_not_zero():
    # one observation -> SEM undefined, must be blank (not 0, which would fake certainty)
    recs = [(date(2023, 1, 2), 9)]
    (row,) = aggregate_by_monday(recs)
    assert row["n_slots"] == 1
    assert row["mean_worms"] == 9.0
    assert row["std_worms"] == ""
    assert row["sem_worms"] == ""


def test_empty_input():
    assert aggregate_by_monday([]) == []


def test_parse_count_handles_int_and_float_strings():
    # nb33 writes "8"; the 2023-2024 pipeline wrote "31.0" -> both are whole worms.
    assert _parse_count("8") == 8
    assert _parse_count("31.0") == 31
    assert isinstance(_parse_count("31.0"), int)
