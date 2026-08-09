"""Unit tests for the pure logic in ``scripts/scene_sampler.py``."""

from __future__ import annotations

import sys
from datetime import UTC, date
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
import scene_sampler as ss


class TestIterSampleDates:
    def test_weekly_mondays_are_all_mondays(self):
        dates = ss.iter_sample_dates(date(2023, 1, 1), date(2024, 12, 31), weekday=0)
        assert all(d.weekday() == 0 for d in dates)

    def test_first_date_is_first_monday_on_or_after_start(self):
        # 2023-01-01 is a Sunday; first Monday is 2023-01-02.
        dates = ss.iter_sample_dates(date(2023, 1, 1), date(2023, 1, 31), weekday=0)
        assert dates[0] == date(2023, 1, 2)

    def test_weekly_spacing_is_seven_days(self):
        dates = ss.iter_sample_dates(date(2023, 1, 1), date(2023, 3, 1), weekday=0)
        gaps = {(b - a).days for a, b in pairwise(dates)}
        assert gaps == {7}

    def test_count_over_two_years_is_about_104(self):
        dates = ss.iter_sample_dates(date(2023, 1, 1), date(2024, 12, 31), weekday=0)
        assert len(dates) == 105  # Mondays 2023-01-02 .. 2024-12-30

    def test_step_weeks_skips(self):
        dates = ss.iter_sample_dates(
            date(2023, 1, 1), date(2023, 3, 1), weekday=0, step_weeks=2
        )
        gaps = {(b - a).days for a, b in pairwise(dates)}
        assert gaps == {14}

    def test_start_on_target_weekday_is_included(self):
        # 2023-01-02 is a Monday.
        dates = ss.iter_sample_dates(date(2023, 1, 2), date(2023, 1, 9), weekday=0)
        assert dates[0] == date(2023, 1, 2)

    def test_empty_range_returns_empty(self):
        assert ss.iter_sample_dates(date(2023, 1, 3), date(2023, 1, 5), weekday=0) == []

    def test_bad_weekday_raises(self):
        with pytest.raises(ValueError):
            ss.iter_sample_dates(date(2023, 1, 1), date(2023, 2, 1), weekday=7)

    def test_bad_step_raises(self):
        with pytest.raises(ValueError):
            ss.iter_sample_dates(date(2023, 1, 1), date(2023, 2, 1), step_weeks=0)


class TestRecordingPath:
    def test_datetime_is_utc(self):
        dt = ss.recording_datetime(date(2024, 10, 17), "121500")
        assert dt.tzinfo == UTC
        assert (dt.hour, dt.minute, dt.second) == (12, 15, 0)

    def test_stem_format(self):
        dt = ss.recording_datetime(date(2024, 10, 17), "121500")
        assert ss.recording_stem(dt) == "CAMHDA301-20241017T121500"

    def test_path_matches_archive_layout(self):
        dt = ss.recording_datetime(date(2024, 10, 17), "121500")
        p = ss.recording_path(Path("/base"), dt)
        assert p == Path("/base/2024/10/17/CAMHDA301-20241017T121500.mp4")

    def test_bad_slot_raises(self):
        with pytest.raises(ValueError):
            ss.recording_datetime(date(2024, 1, 1), "1215")

    def test_all_slots_are_valid_six_digit(self):
        assert len(ss.SLOTS) == 8
        for slot in ss.SLOTS:
            assert len(slot) == 6 and slot.isdigit()


class TestFrameOffsets:
    def test_spacing_and_bounds(self):
        offs = ss.frame_offsets(duration_s=891.0, every_s=30.0)
        assert offs[0] == 0.0
        assert offs[1] == 30.0
        assert offs[-1] <= 891.0 - 1.0

    def test_count_for_known_clip(self):
        # 0,30,...,870 -> 30 frames (890 - 1 pad = 889 limit; 870 <= 889 < 900).
        offs = ss.frame_offsets(duration_s=890.0, every_s=30.0)
        assert len(offs) == 30

    def test_zero_duration_is_empty(self):
        assert ss.frame_offsets(0.0, 30.0) == []

    def test_bad_every_raises(self):
        with pytest.raises(ValueError):
            ss.frame_offsets(100.0, 0.0)


class TestGridShape:
    @pytest.mark.parametrize(
        "n,cols,expected",
        [(30, 6, (5, 6)), (29, 6, (5, 6)), (31, 6, (6, 6)), (1, 6, (1, 6))],
    )
    def test_rows(self, n, cols, expected):
        assert ss.grid_shape(n, cols) == expected

    def test_zero_tiles(self):
        assert ss.grid_shape(0, 6) == (0, 6)

    def test_bad_cols_raises(self):
        with pytest.raises(ValueError):
            ss.grid_shape(10, 0)


class TestEnumerateTargets:
    def test_count_is_days_times_slots(self):
        dates = [date(2024, 10, 7), date(2024, 10, 14)]
        targets = ss.enumerate_targets(Path("/base"), dates)
        assert len(targets) == len(dates) * len(ss.SLOTS)

    def test_target_stem_roundtrips(self):
        targets = ss.enumerate_targets(Path("/base"), [date(2024, 10, 17)], ["121500"])
        assert targets[0].stem == "CAMHDA301-20241017T121500"


class TestClampWorkers:
    def test_clamps_above_ceiling(self):
        assert ss.clamp_workers(100) == ss.MAX_WORKERS

    def test_floor_is_one(self):
        assert ss.clamp_workers(0) == 1

    def test_passthrough_within_range(self):
        assert ss.clamp_workers(8) == 8
