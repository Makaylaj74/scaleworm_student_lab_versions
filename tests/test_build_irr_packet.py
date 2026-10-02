"""Tests for scripts/build_irr_packet.py (blind IRR packet builder)."""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_irr_packet as bip


def test_pick_even_all_when_k_ge_n():
    assert bip.pick_even(["a", "b", "c"], 4) == ["a", "b", "c"]


def test_pick_even_endpoints_included():
    got = bip.pick_even(["0", "1", "2", "3", "4", "5", "6", "7"], 4)
    assert got[0] == "0" and got[-1] == "7"
    assert len(got) == 4


def test_pick_even_one():
    assert bip.pick_even(["a", "b", "c"], 1) == ["a"]


def test_blank_manifest_rows():
    fields = ["frame_id", "datetime_utc", "split", "frame_status", "worm_count",
              "label_path", "prelabel_model", "counter", "annotation_method"]
    rows = [{"frame_id": "CAMHDA301-20220117T001500", "datetime_utc": "x",
             "split": "train", "frame_status": "counted", "worm_count": "9",
             "label_path": "labels/old.txt", "prelabel_model": "v2",
             "counter": "MJ", "annotation_method": "box_corrected"}]
    out = bip.blank_manifest_rows(rows, fields)
    r = out[0]
    assert r["frame_id"] == "CAMHDA301-20220117T001500"  # provenance kept
    assert r["split"] == "train"
    assert r["frame_status"] == "pending"  # reset
    assert r["worm_count"] == ""  # blanked
    assert r["counter"] == ""
    assert r["annotation_method"] == ""
    assert r["prelabel_model"] == "none"  # blind
    assert r["label_path"] == "labels/CAMHDA301-20220117T001500.txt"


def test_build_notebook_structure():
    nb = bip.build_notebook("LG", 12)
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 4
    assert nb["cells"][0]["cell_type"] == "markdown"
    # self-contained setup cell installs ipympl so no special kernel is needed
    setup = "".join(nb["cells"][1]["source"])
    assert "ipympl" in setup
    config = "".join(nb["cells"][2]["source"])
    assert 'irr_packet_LG' in config
    assert 'COUNTER = "LG"' in config
    # generic default kernel, not a personal/shared one
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert "joseph-scaleworm-thesis" not in json.dumps(nb)
    # the whole notebook must be JSON-serialisable
    json.dumps(nb)


def _fixture_series(tmp_path, days_frames):
    """Build a fake monday_manual_series with a manifest + blank png images."""
    series = tmp_path / "series"
    images = series / "images"
    images.mkdir(parents=True)
    fields = ["frame_id", "datetime_utc", "camera_unit", "quarter", "scene1_time_s",
              "video_path", "sharpness", "split", "frame_status", "worm_count",
              "label_path", "annotation_method", "prelabel_model", "counter",
              "date_counted", "notes"]
    rows = []
    for fid, status in days_frames:
        (images / f"{fid}.png").write_bytes(b"\x89PNG\r\n")  # dummy image bytes
        rows.append({k: "" for k in fields} | {
            "frame_id": fid, "split": "train", "frame_status": status,
            "worm_count": "7" if status == "counted" else ""})
    manifest = series / "frame_manifest.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return series, images, manifest


def test_select_frames_only_counted(tmp_path, monkeypatch):
    days_frames = [
        ("CAMHDA301-20220117T001500", "counted"),
        ("CAMHDA301-20220117T031500", "counted"),
        ("CAMHDA301-20220131T001500", "no_scene1"),   # excluded (not counted)
        ("CAMHDA301-20230109T001500", "counted"),
        ("CAMHDA301-20230601T001500", "counted"),       # not in a wanted day
    ]
    _series, images, manifest = _fixture_series(tmp_path, days_frames)
    monkeypatch.setattr(bip, "MANIFEST", manifest)
    monkeypatch.setattr(bip, "IMAGES", images)
    monkeypatch.setattr(bip, "PERIODS",
                        {"low": ["20220117", "20220131"], "peak": ["20230109"]})
    sel = bip.select_frames(slots_per_day=4)
    ids = {r["frame_id"] for r in sel}
    assert "CAMHDA301-20220117T001500" in ids
    assert "CAMHDA301-20220131T001500" not in ids  # no_scene1
    assert "CAMHDA301-20230601T001500" not in ids  # wrong day
    assert len(sel) == 3


def test_build_packet_end_to_end(tmp_path, monkeypatch):
    days_frames = [
        ("CAMHDA301-20220117T001500", "counted"),
        ("CAMHDA301-20220117T031500", "counted"),
        ("CAMHDA301-20230109T001500", "counted"),
    ]
    _series, images, manifest = _fixture_series(tmp_path, days_frames)
    monkeypatch.setattr(bip, "MANIFEST", manifest)
    monkeypatch.setattr(bip, "IMAGES", images)
    monkeypatch.setattr(bip, "PERIODS",
                        {"low": ["20220117"], "peak": ["20230109"]})
    out = tmp_path / "packets"
    zip_path = bip.build_packet("LG", slots_per_day=4, out_dir=out)

    assert zip_path.exists()
    pkt = out / "irr_packet_LG"
    # images copied, labels empty, notebook + readme present
    assert (pkt / "frames" / "CAMHDA301-20220117T001500.png").exists()
    assert (pkt / "labeler.ipynb").exists()
    assert (pkt / "README.md").exists()
    # manifest blanked
    mrows = list(csv.DictReader((pkt / "irr_manifest.csv").open(newline="")))
    assert all(r["frame_status"] == "pending" for r in mrows)
    assert all(r["worm_count"] == "" for r in mrows)
    assert len(mrows) == 3
    # zip contains the manifest
    with zipfile.ZipFile(zip_path) as zf:
        assert "irr_packet_LG/irr_manifest.csv" in zf.namelist()


def test_build_packet_missing_image_raises(tmp_path, monkeypatch):
    days_frames = [("CAMHDA301-20220117T001500", "counted")]
    _series, images, manifest = _fixture_series(tmp_path, days_frames)
    (images / "CAMHDA301-20220117T001500.png").unlink()  # remove the only image
    monkeypatch.setattr(bip, "MANIFEST", manifest)
    monkeypatch.setattr(bip, "IMAGES", images)
    monkeypatch.setattr(bip, "PERIODS", {"low": ["20220117"]})
    with pytest.raises(FileNotFoundError):
        bip.build_packet("LG", slots_per_day=4, out_dir=tmp_path / "p")
