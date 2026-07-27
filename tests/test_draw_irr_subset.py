"""Unit tests for ``scripts/draw_irr_subset.py``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import draw_irr_subset as draw
import pandas as pd

SLOTS = ("001500", "031500", "061500", "091500", "121500", "151500", "181500", "211500")


def _fake_manifest(n_per_stratum: int = 6) -> pd.DataFrame:
    """Two years x eight slots, n recordings each, plus a few non-existent videos."""
    rows = []
    for year in (2023, 2024):
        for slot in SLOTS:
            for day in range(1, n_per_stratum + 1):
                stem = f"CAMHDA301-{year}{day:02d}01T{slot}"
                rows.append(
                    {
                        "stem": stem,
                        "utc_datetime": f"{year}-{day:02d}-01T"
                        f"{slot[:2]}:{slot[2:4]}:00+00:00",
                        "weekday": "Monday",
                        "slot": slot,
                        "video_exists": True,
                        "sheet_path": f"/x/{stem}.png",
                    }
                )
    # one row whose video does not exist — must be excluded
    rows.append(
        {
            "stem": "CAMHDA301-20230801T001500",
            "utc_datetime": "2023-08-01T00:15:00+00:00",
            "weekday": "Monday",
            "slot": "001500",
            "video_exists": False,
            "sheet_path": "/x/missing.png",
        }
    )
    return pd.DataFrame(rows)


def test_allocate_sums_and_evenness():
    quotas = draw.allocate(80, 16)
    assert sum(quotas) == 80
    assert max(quotas) - min(quotas) <= 1


def test_allocate_remainder_goes_first():
    assert draw.allocate(10, 4) == [3, 3, 2, 2]


def test_load_manifest_excludes_nonexistent_and_adds_year(tmp_path):
    mpath = tmp_path / "manifest.csv"
    _fake_manifest().to_csv(mpath, index=False)
    df = draw.load_manifest(mpath)
    assert (df["video_exists"].astype(str).str.lower() == "true").all()
    assert set(df["year"].unique()) == {2023, 2024}
    assert not (df["stem"] == "CAMHDA301-20230801T001500").any()


def test_draw_subset_is_deterministic():
    manifest = _fake_manifest()
    manifest["year"] = pd.to_datetime(manifest["utc_datetime"], utc=True).dt.year
    manifest = manifest[manifest["video_exists"]].reset_index(drop=True)
    a = draw.draw_subset(manifest, n_target=32, seed=1)
    b = draw.draw_subset(manifest, n_target=32, seed=1)
    c = draw.draw_subset(manifest, n_target=32, seed=2)
    assert list(a["stem"]) == list(b["stem"])
    assert list(a["stem"]) != list(c["stem"])  # different seed -> different draw


def test_draw_subset_stratified_coverage():
    manifest = _fake_manifest()
    manifest["year"] = pd.to_datetime(manifest["utc_datetime"], utc=True).dt.year
    subset = draw.draw_subset(manifest, n_target=32, seed=7)
    # 16 strata, 32 target -> 2 per stratum, all present
    assert len(subset) == 32
    assert subset["stratum"].nunique() == 16
    assert (subset["stratum"].value_counts() == 2).all()
    assert subset["stem"].is_unique


def test_draw_subset_caps_at_available():
    manifest = _fake_manifest(n_per_stratum=1)  # only 1 per stratum -> 16 available
    manifest["year"] = pd.to_datetime(manifest["utc_datetime"], utc=True).dt.year
    manifest = manifest[manifest["video_exists"]].reset_index(drop=True)
    subset = draw.draw_subset(manifest, n_target=100, seed=3)
    assert len(subset) == 16  # cannot exceed what exists; no duplicates
    assert subset["stem"].is_unique


def test_find_sheet_searches_sorted_subdirs(tmp_path):
    session = tmp_path / "session"
    (session / "not_scene1").mkdir(parents=True)
    (session / "not_scene1" / "CAMHDA301-X.png").write_bytes(b"png")
    assert draw.find_sheet("CAMHDA301-X", session).name == "CAMHDA301-X.png"
    assert draw.find_sheet("CAMHDA301-absent", session) is None


def test_build_rater_dirs_copies_and_reports_missing(tmp_path):
    # source session with sheets for two of three selected stems
    session = tmp_path / "full"
    (session / "contact_sheets").mkdir(parents=True)
    (session / "scene1").mkdir()
    (session / "not_scene1").mkdir()
    (session / "contact_sheets" / "CAMHDA301-A.png").write_bytes(b"png")
    (session / "scene1" / "CAMHDA301-B.png").write_bytes(b"png")  # already sorted

    subset = pd.DataFrame(
        {
            "stem": ["CAMHDA301-A", "CAMHDA301-B", "CAMHDA301-C"],
            "stratum": ["2023_001500"] * 3,
        }
    )
    out = tmp_path / "irr_round_1"
    summary = draw.build_rater_dirs(subset, out, session, ["rater_a", "rater_b"])

    assert summary["copied"] == ["CAMHDA301-A", "CAMHDA301-B"]
    assert summary["missing"] == ["CAMHDA301-C"]
    assert (out / "irr_subset.csv").is_file()
    for rater in ("rater_a", "rater_b"):
        sheets = out / "raters" / rater / "contact_sheets"
        assert (sheets / "CAMHDA301-A.png").is_file()
        assert (sheets / "CAMHDA301-B.png").is_file()
        assert not (sheets / "CAMHDA301-C.png").exists()
        assert (out / "raters" / rater / "scene1").is_dir()
