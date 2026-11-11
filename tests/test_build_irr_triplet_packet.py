"""Tests for scripts/build_irr_triplet_packet.py (round-4 before/at/after packets)."""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_irr_triplet_packet as bt

FIELDS = [
    "frame_id",
    "datetime_utc",
    "scene1_time_s",
    "video_path",
    "sharpness",
    "split",
    "frame_status",
    "worm_count",
    "label_path",
    "annotation_method",
    "prelabel_model",
    "counter",
    "date_counted",
    "notes",
    *bt.EXTRA_FIELDS,
]


def _row(fid, count="10", t="210", day="2022-09-19"):
    return {
        "frame_id": fid,
        "datetime_utc": f"{day}T21:15:00",
        "scene1_time_s": t,
        "video_path": "/v.mp4",
        "sharpness": "150",
        "split": "train",
        "frame_status": "counted",
        "worm_count": count,
    }


def test_neighbour_id_sorts_and_reads():
    assert bt.neighbour_id("CAM-20220919T211500", -10) == "CAM-20220919T211500_dt-10"
    assert bt.neighbour_id("CAM-20220919T211500", 0) == "CAM-20220919T211500_dt+00"
    assert bt.neighbour_id("CAM-20220919T211500", 10) == "CAM-20220919T211500_dt+10"


def test_triplet_rows_order_times_and_blanks():
    rows = bt.triplet_rows([_row("CAM-A"), _row("CAM-B", t="420")], FIELDS, offset_s=10)
    assert [r["frame_id"] for r in rows] == [
        "CAM-A_dt-10",
        "CAM-A_dt+00",
        "CAM-A_dt+10",
        "CAM-B_dt-10",
        "CAM-B_dt+00",
        "CAM-B_dt+10",
    ]
    assert [r["scene1_time_s"] for r in rows[3:]] == ["410", "420", "430"]
    assert [r["triplet_id"] for r in rows] == ["T1"] * 3 + ["T2"] * 3
    assert all(r["frame_status"] == "pending" and r["worm_count"] == "" for r in rows)
    assert all(r["prelabel_model"] == "none" for r in rows)
    assert rows[0]["label_path"] == "labels/CAM-A_dt-10.txt"


def test_phase_shift_recovers_translation():
    rng = np.random.default_rng(0)
    a = rng.random((64, 80))
    b = np.roll(a, (3, -5), axis=(0, 1))
    assert bt.phase_shift(b, a) == (3, -5)


def test_sharpness_drops_with_blur():
    rng = np.random.default_rng(1)
    sharp = rng.random((90, 90))
    blur = (
        sharp
        + np.roll(sharp, 1, 0)
        + np.roll(sharp, 1, 1)
        + np.roll(sharp, (1, 1), (0, 1))
    ) / 4
    assert bt.sharpness(blur) < 0.5 * bt.sharpness(sharp)


def _write_rgb(path, gray):
    mpimg.imsave(path, np.stack([gray] * 3, axis=-1))


def test_check_triplet_flags_blur_and_motion(tmp_path):
    rng = np.random.default_rng(2)
    base = rng.random((120, 160))
    blur = (
        base + np.roll(base, 1, 0) + np.roll(base, 1, 1) + np.roll(base, (1, 1), (0, 1))
    ) / 4
    for name, g in {"a": base, "b": base, "c": blur, "d": np.roll(base, 20, 1)}.items():
        _write_rgb(tmp_path / f"{name}.png", g)
    p = {k: tmp_path / f"{k}.png" for k in "abcd"}
    assert bt.check_triplet([p["a"], p["b"], p["a"]], step=1)[0]
    assert not bt.check_triplet([p["a"], p["b"], p["c"]], step=1)[0]  # focus mismatch
    assert not bt.check_triplet([p["d"], p["b"], p["a"]], step=1)[0]  # camera moved


def test_notebook_has_triplet_view_and_packet_root():
    nb = bt.build_notebook("LG", "irr_packet_LG_round4", 15)
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert 'ROOT = Path.home() / "irr_packet_LG_round4"' in src
    assert "class TripletLabeler(BlindLabeler)" in src
    assert "app = TripletLabeler(stems)" in src
    assert "app = BlindLabeler(stems)" not in src
    assert "triplets" in "".join(nb["cells"][0]["source"])


def test_build_packet_end_to_end(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = bt.triplet_rows([_row("CAM-A")], FIELDS)
    for r in rows:
        _write_rgb(bt.cached(r, cache), np.zeros((8, 8)))
    zp = bt.build_packet("MS", rows, FIELDS, tmp_path, cache=cache)
    pkt = tmp_path / "irr_packet_MS_round4"
    assert sorted(p.name for p in (pkt / "frames").iterdir()) == sorted(
        f"{r['frame_id']}.png" for r in rows
    )
    got = list(csv.DictReader((pkt / "irr_manifest.csv").open()))
    assert [r["offset_s"] for r in got] == ["-10", "0", "10"]
    readme = (pkt / "README.md").read_text()
    assert "irr_packet_MS_round4" in readme and "irr_packet_MS.zip" not in readme
    assert "groups of 3" in readme
    json.loads((pkt / "labeler.ipynb").read_text())
    with zipfile.ZipFile(zp) as zf:
        assert "irr_packet_MS_round4/irr_manifest.csv" in zf.namelist()


def test_seen_days_scans_names_and_text_but_skips_exempt(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "CAMHDA301-20220101T001500.png").write_text("")
    (tmp_path / "a" / "log.csv").write_text("x,CAMHDA301-20220202T031500\n")
    (tmp_path / "own").mkdir()
    (tmp_path / "own" / "CAMHDA301-20220303T001500.png").write_text("")
    monkeypatch.setattr(bt, "EXEMPT", [tmp_path / "own"])
    assert bt.seen_days(["a", "own"], root=tmp_path) == {"20220101", "20220202"}


def test_anchor_candidates_weekday_window_unseen_only():
    def h(fid, dens):
        d = fid[10:18]
        return {
            "frame_id": fid,
            "datetime_utc": f"{d[:4]}-{d[4:6]}-{d[6:]}T00:15:00",
            "_density": dens,
        }

    rows = [
        h("CAMHDA301-20220103T001500", 1),  # Monday -> out
        h("CAMHDA301-20220104T001500", 2),  # Tuesday, seen day -> out
        h("CAMHDA301-20220105T001500", 3),  # Wednesday -> in
        h("CAMHDA301-20220108T001500", 4),  # Saturday -> out
        h("CAMHDA301-20240105T001500", 5),  # outside clear window -> out
    ]
    cands = bt.anchor_candidates(rows, excluded_days={"20220104"})
    assert {c["frame_id"] for q in cands for c in q} == {"CAMHDA301-20220105T001500"}
