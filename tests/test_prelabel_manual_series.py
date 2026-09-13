"""Tests for the manifest-driven pre-label router (pure logic; no model/ffmpeg)."""

import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prelabel_manual_series as pms


def test_clear_frame_routes_to_v2():
    assert pms.model_for_date(datetime(2023, 1, 2, 0, 15)) == "v2"
    assert pms.model_for_date(datetime(2022, 6, 1)) == "v2"


def test_blurry_frame_routes_to_v3():
    assert pms.model_for_date(datetime(2024, 5, 6, 9, 15)) == "v3"
    assert pms.model_for_date(datetime(2023, 12, 25)) == "v3"


def test_boundary_date_is_inclusive_to_v3():
    # On the swap date itself the footage is already the new (blurry) unit.
    assert pms.model_for_date(datetime(2023, 8, 10, 0, 0)) == "v3"
    assert pms.model_for_date(datetime(2023, 8, 9, 21, 15)) == "v2"


def test_custom_blur_onset():
    onset = date(2024, 1, 1)
    assert pms.model_for_date(datetime(2023, 12, 31), onset) == "v2"
    assert pms.model_for_date(datetime(2024, 1, 1), onset) == "v3"


def test_targets_skips_protected_and_unpicked_and_done(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "CAMHDA301-20230109T001500.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    rows = [
        {"frame_id": "CAMHDA301-20230102T001500", "scene1_time_s": "210", "frame_status": "pending"},
        {"frame_id": "CAMHDA301-20230109T001500", "scene1_time_s": "300", "frame_status": "prelabeled"},  # done
        {"frame_id": "CAMHDA301-20230116T001500", "scene1_time_s": "", "frame_status": "pending"},  # unpicked
        {"frame_id": "CAMHDA301-20230123T001500", "scene1_time_s": "210", "frame_status": "counted"},  # human-owned
    ]
    todo = pms._targets(rows, labels, force=False)
    assert [r["frame_id"] for r in todo] == ["CAMHDA301-20230102T001500"]


def test_targets_force_redoes_existing_but_still_skips_protected(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "CAMHDA301-20230109T001500.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    rows = [
        {"frame_id": "CAMHDA301-20230109T001500", "scene1_time_s": "300", "frame_status": "prelabeled"},
        {"frame_id": "CAMHDA301-20230123T001500", "scene1_time_s": "210", "frame_status": "counted"},
    ]
    todo = pms._targets(rows, labels, force=True)
    assert [r["frame_id"] for r in todo] == ["CAMHDA301-20230109T001500"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
