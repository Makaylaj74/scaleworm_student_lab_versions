"""Tests for the manual-series frame-manifest builder."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

import build_frame_manifest as bfm


def test_stem_to_dt():
    assert bfm.stem_to_dt("CAMHDA301-20230102T001500") == datetime(2023, 1, 2, 0, 15, 0)


def test_quarter():
    assert bfm.quarter(datetime(2023, 1, 2)) == "2023-Q1"
    assert bfm.quarter(datetime(2023, 12, 30)) == "2023-Q4"
    assert bfm.quarter(datetime(2024, 4, 1)) == "2024-Q2"


def test_split_all_slots_of_a_day_share_a_split():
    # All eight 3-hourly slots on one Monday must land in the same split,
    # otherwise near-duplicate same-day frames leak across train/val.
    day = datetime(2023, 1, 2)
    splits = {bfm.split_for(day.replace(hour=h)) for h in (0, 3, 6, 9, 12, 15, 18, 21)}
    assert len(splits) == 1


def test_split_is_stable_per_date():
    # Same date -> same split regardless of anything else (append-safe).
    d = datetime(2024, 5, 6, 9, 15)
    assert bfm.split_for(d) == bfm.split_for(d)


def test_split_ratio_is_about_one_in_five():
    # Over the real 2023-2024 Monday grid, ~20% should be val.
    mondays = [datetime(2023, 1, 2)]
    while mondays[-1] < datetime(2024, 12, 30):
        mondays.append(mondays[-1] + timedelta(days=7))
    frac = sum(bfm.split_for(m) == "val" for m in mondays) / len(mondays)
    assert 0.15 <= frac <= 0.25


def test_split_values_are_valid():
    assert bfm.split_for(datetime(2023, 1, 2)) in {"train", "val"}


def test_video_path():
    p = bfm.video_path("CAMHDA301-20230102T001500", datetime(2023, 1, 2, 0, 15))
    assert p.endswith("/2023/01/02/CAMHDA301-20230102T001500.mp4")


def test_build_row_new_frame_is_pending():
    row = bfm.build_row("CAMHDA301-20230102T001500", "210", {}, None)
    assert row["frame_status"] == "pending"
    assert row["worm_count"] == ""
    assert row["scene1_time_s"] == "210"


def test_build_row_preserves_prior_annotations():
    prior = {c: "" for c in bfm.COLUMNS}
    prior.update(worm_count="12", frame_status="counted", counter="MJ")
    row = bfm.build_row("CAMHDA301-20230102T001500", "210", {}, prior)
    assert row["worm_count"] == "12"
    assert row["frame_status"] == "counted"
    assert row["counter"] == "MJ"
    # source columns are still refreshed
    assert row["quarter"] == "2023-Q1"


def test_read_scene1_stems_filters_not_scene1(tmp_path):
    log = tmp_path / "sort_log.csv"
    log.write_text(
        "stem,decision,scene1_time_s,decided_at_utc\n"
        "CAMHDA301-20230102T001500,scene1,210,x\n"
        "CAMHDA301-20230102T031500,not_scene1,,x\n"
    )
    rows = bfm.read_scene1_stems([log])
    assert rows == [("CAMHDA301-20230102T001500", "210")]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
