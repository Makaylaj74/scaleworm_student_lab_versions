"""Unit tests for ``scripts/scene_sorter.py``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
import scene_sorter as sorter


def _make_session(tmp_path: Path, stems: list[str]) -> Path:
    """Create a session dir with fake contact-sheet PNGs."""
    sheets = tmp_path / "contact_sheets"
    sheets.mkdir(parents=True)
    for stem in stems:
        (sheets / f"{stem}.png").write_bytes(b"fake png")
    return tmp_path


class TestPending:
    def test_lists_all_sheets_sorted(self, tmp_path):
        s = _make_session(tmp_path, ["c", "a", "b"])
        assert [p.stem for p in sorter.pending_sheets(s)] == ["a", "b", "c"]

    def test_empty_when_none(self, tmp_path):
        sorter.session_paths(tmp_path)  # creates empty dirs
        assert sorter.pending_sheets(tmp_path) == []


class TestApplyDecision:
    def test_scene1_moves_and_returns_dest(self, tmp_path):
        s = _make_session(tmp_path, ["x"])
        sheet = sorter.pending_sheets(s)[0]
        dest = sorter.apply_decision(s, sheet, "scene1")
        assert dest is not None
        assert dest.parent.name == "scene1"
        assert dest.is_file()
        assert not sheet.exists()  # moved, not copied
        assert sorter.pending_sheets(s) == []

    def test_not_scene1_moves_to_correct_folder(self, tmp_path):
        s = _make_session(tmp_path, ["y"])
        dest = sorter.apply_decision(s, sorter.pending_sheets(s)[0], "not_scene1")
        assert dest.parent.name == "not_scene1"

    def test_skip_leaves_file_and_returns_none(self, tmp_path):
        s = _make_session(tmp_path, ["z"])
        sheet = sorter.pending_sheets(s)[0]
        assert sorter.apply_decision(s, sheet, "skip") is None
        assert sheet.exists()  # still pending

    def test_decision_is_logged(self, tmp_path):
        s = _make_session(tmp_path, ["a", "b"])
        sorter.apply_decision(s, s / "contact_sheets" / "a.png", "scene1")
        sorter.apply_decision(s, s / "contact_sheets" / "b.png", "skip")
        rows = (s / "sort_log.csv").read_text().splitlines()
        assert rows[0] == "stem,decision,scene1_time_s,decided_at_utc"
        assert rows[1].startswith("a,scene1,,")  # no time given -> blank column
        assert rows[2].startswith("b,skip,,")

    def test_overwrites_existing_destination(self, tmp_path):
        s = _make_session(tmp_path, ["dup"])
        (s / "scene1").mkdir(exist_ok=True)
        (s / "scene1" / "dup.png").write_bytes(b"old")
        dest = sorter.apply_decision(s, s / "contact_sheets" / "dup.png", "scene1")
        assert dest.read_bytes() == b"fake png"

    def test_bad_decision_raises(self, tmp_path):
        s = _make_session(tmp_path, ["a"])
        with pytest.raises(ValueError):
            sorter.apply_decision(s, sorter.pending_sheets(s)[0], "maybe")

    def test_missing_file_raises(self, tmp_path):
        sorter.session_paths(tmp_path)
        with pytest.raises(FileNotFoundError):
            sorter.apply_decision(
                tmp_path, tmp_path / "contact_sheets" / "nope.png", "scene1"
            )

    def test_explicit_timestamp_recorded(self, tmp_path):
        s = _make_session(tmp_path, ["a"])
        sorter.apply_decision(
            s,
            s / "contact_sheets" / "a.png",
            "scene1",
            when="2026-07-27T12:00:00+00:00",
        )
        assert "2026-07-27T12:00:00+00:00" in (s / "sort_log.csv").read_text()

    def test_scene1_time_recorded_in_log(self, tmp_path):
        s = _make_session(tmp_path, ["a"])
        sorter.apply_decision(
            s, s / "contact_sheets" / "a.png", "scene1", scene1_time_s=270
        )
        rows = (s / "sort_log.csv").read_text().splitlines()
        assert rows[1].startswith("a,scene1,270,")

    def test_scene1_time_on_non_scene1_raises(self, tmp_path):
        s = _make_session(tmp_path, ["a"])
        with pytest.raises(ValueError):
            sorter.apply_decision(
                s, s / "contact_sheets" / "a.png", "not_scene1", scene1_time_s=270
            )

    def test_negative_scene1_time_raises(self, tmp_path):
        s = _make_session(tmp_path, ["a"])
        with pytest.raises(ValueError):
            sorter.apply_decision(
                s, s / "contact_sheets" / "a.png", "scene1", scene1_time_s=-5
            )


class TestCounts:
    def test_counts_track_moves(self, tmp_path):
        s = _make_session(tmp_path, ["a", "b", "c"])
        assert sorter.counts(s) == {"scene1": 0, "not_scene1": 0, "pending": 3}
        sorter.apply_decision(s, s / "contact_sheets" / "a.png", "scene1")
        sorter.apply_decision(s, s / "contact_sheets" / "b.png", "not_scene1")
        assert sorter.counts(s) == {"scene1": 1, "not_scene1": 1, "pending": 1}
