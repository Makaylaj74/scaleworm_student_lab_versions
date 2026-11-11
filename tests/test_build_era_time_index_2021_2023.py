"""Tests for the era-time automated index: era rules and weekly aggregation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_era_time_index_2021_2023 import aggregate, era_times


@pytest.mark.parametrize(
    ("day", "times"),
    [
        ("20210104", (420,)),
        ("20210831", (420,)),
        ("20210901", (540,)),
        ("20220424", (540,)),
        ("20220425", (210,)),
        ("20221231", (210,)),
        ("20230101", (210, 420)),
        ("20230531", (210, 420)),
        ("20230601", (210,)),
    ],
)
def test_era_times(day, times):
    assert era_times(day) == times


def _reads(rows):
    return pd.DataFrame(rows, columns=["frame_id", "worm_index"])


def test_aggregate_drops_offscene_and_averages_days_then_week():
    reads = _reads(
        [
            ("CAMHDA301-20210104T001500", 10.0),  # Mon
            ("CAMHDA301-20210104T031500", 20.0),  # Mon -> day mean 15
            ("CAMHDA301-20210104T061500", 1.0),  # off-scene, dropped
            ("CAMHDA301-20210105T001500", 25.0),  # Tue -> day mean 25
        ]
    )
    wk = aggregate(reads, cutoff=8.5)
    assert len(wk) == 1
    row = wk.iloc[0]
    assert wk.index[0] == pd.Timestamp("2021-01-04")
    assert row.worm_index == pytest.approx(20.0)  # mean of days, not of recordings
    assert row.n_days == 2
    assert row.n_recordings == 3
    assert row.n_offscene == 1


def test_aggregate_week_with_all_offscene_is_absent():
    reads = _reads(
        [("CAMHDA301-20210104T001500", 12.0), ("CAMHDA301-20210111T001500", 2.0)]
    )
    wk = aggregate(reads, cutoff=8.5)
    assert list(wk.index) == [pd.Timestamp("2021-01-04")]
