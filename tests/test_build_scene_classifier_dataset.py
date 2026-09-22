"""Tests for the Scene-1 classifier dataset assembler."""

import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_scene_classifier_dataset as b


def test_stem_parsing_and_video_path():
    assert b.stem_to_dt("CAMHDA301-20230808T061500") == datetime(2023, 8, 8, 6, 15, 0)  # noqa: DTZ001
    vp = b.stem_to_video("CAMHDA301-20230808T061500", base=Path("/base"))
    assert vp == Path("/base/2023/08/08/CAMHDA301-20230808T061500.mp4")


def _write_logs(tmp_path):
    """Two batches: a primary and a validation re-label that both cover one stem,
    plus an ambiguous stem (disagreeing decisions)."""

    def w(batch, rows):
        d = tmp_path / batch
        d.mkdir()
        with (d / "sort_log.csv").open("w", newline="") as f:
            wr = csv.DictWriter(
                f, fieldnames=["stem", "decision", "scene1_time_s", "decided_at_utc"]
            )
            wr.writeheader()
            wr.writerows(rows)

    w(
        "full_2023_2024",
        [
            {
                "stem": "CAMHDA301-20230307T001500",
                "decision": "scene1",
                "scene1_time_s": "540",
                "decided_at_utc": "x",
            },
            {
                "stem": "CAMHDA301-20230307T031500",
                "decision": "not_scene1",
                "scene1_time_s": "",
                "decided_at_utc": "x",
            },
            {
                "stem": "CAMHDA301-20230314T001500",
                "decision": "scene1",
                "scene1_time_s": "300",
                "decided_at_utc": "x",
            },
            {
                "stem": "CAMHDA301-20230321T001500",
                "decision": "skip",
                "scene1_time_s": "",
                "decided_at_utc": "x",
            },
        ],
    )
    # validation batch re-labels 0307T001500 (agrees) and 0314T001500 (DISAGREES)
    w(
        "validation_2023_03",
        [
            {
                "stem": "CAMHDA301-20230307T001500",
                "decision": "scene1",
                "scene1_time_s": "540",
                "decided_at_utc": "y",
            },
            {
                "stem": "CAMHDA301-20230314T001500",
                "decision": "not_scene1",
                "scene1_time_s": "",
                "decided_at_utc": "y",
            },
        ],
    )
    return str(tmp_path / "**/sort_log.csv")


def test_dedup_prefers_primary_and_drops_disagreements(tmp_path):
    g = _write_logs(tmp_path)
    dec = b.load_decisions(g)
    dropped = dec.pop("_dropped_ambiguous")
    # 0314 disagrees across passes -> dropped; skip dropped; 0307 deduped to primary
    assert dropped == 1
    assert "CAMHDA301-20230314T001500" not in dec
    assert "CAMHDA301-20230321T001500" not in dec  # skip
    assert dec["CAMHDA301-20230307T001500"]["batch"] == "full_2023_2024"
    assert set(dec) == {"CAMHDA301-20230307T001500", "CAMHDA301-20230307T031500"}


def test_labeling_rule_positive_and_negatives(tmp_path):
    g = _write_logs(tmp_path)
    dec = b.load_decisions(g)
    dec.pop("_dropped_ambiguous")
    rows = b.build_rows(dec, neg_per=3, seed=1)
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    # one positive at the annotated time
    assert len(pos) == 1
    assert pos[0]["tile_time_s"] == 540
    assert pos[0]["stem"] == "CAMHDA301-20230307T001500"
    # exactly neg_per negatives for the one not_scene1 recording, all off the grid
    assert len(neg) == 3
    assert all(
        30 <= r["tile_time_s"] <= 840 and r["tile_time_s"] % 30 == 0 for r in neg
    )
    assert all(r["tile_time_s"] != 0 for r in neg)


def test_day_split_deterministic_and_no_leakage(tmp_path):
    g = _write_logs(tmp_path)
    dec = b.load_decisions(g)
    dec.pop("_dropped_ambiguous")
    rows = b.build_rows(dec, neg_per=4, seed=7)
    # a date maps to a single split, stably
    day_to_splits = {}
    for r in rows:
        day_to_splits.setdefault(r["date"], set()).add(r["split"])
    assert all(len(s) == 1 for s in day_to_splits.values())
    # deterministic
    assert b.day_split("2023-03-07", 7) == b.day_split("2023-03-07", 7)


def test_era_boundary():
    # blur onset is 2023-08-10
    dec = {
        "CAMHDA301-20230808T001500": {
            "batch": "b",
            "decision": "scene1",
            "scene1_time_s": "300",
        },
        "CAMHDA301-20230815T001500": {
            "batch": "b",
            "decision": "scene1",
            "scene1_time_s": "300",
        },
    }
    rows = {r["stem"]: r for r in b.build_rows(dec, neg_per=1, seed=1)}
    assert rows["CAMHDA301-20230808T001500"]["era"] == "clear"
    assert rows["CAMHDA301-20230815T001500"]["era"] == "blurry"
