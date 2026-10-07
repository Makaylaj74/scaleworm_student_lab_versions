"""Build a density-stratified hand-count subsample for the 2015-2016 post-eruption era.

2015-2016 is the core scientific motivation but the automated detector is out-of-distribution there
(dense tubeworm mound, not the 2021-era tower; ~28% recall) and there is no manual truth. To validate,
anchor, and decide the measurement metric (individual count vs percent-cover -- advisor-blocked), we
need a hand-count subsample that SPANS THE DENSITY RANGE, not a random/time-only sample.

Pool: the Scene-1 frames already identified by the manual scene sort (scene_sorting/full_2015 = 120,
scene_sorting/subsample_2016 = 18), each with a scene1_time_s and a video_path. Steps:
  1. Extract each Scene-1 frame and score it with v8+SAHI (a ROUGH density proxy -- the detector is OOD
     here so counts are only an ordering, used to stratify, not as truth). Resumable (cached scores).
  2. Stratify-pick TARGET frames across detector-density terciles x time, so sparse, moderate and dense
     scenes are all represented and the 2015-2016 recovery timeline is spanned.
  3. Render a hand-count packet (frames + blank sort_sheet + README) for manual counting.

Run (thesis venv: sahi + cuda):
  /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python scripts/build_subsample_2015_2016.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_frames import extract_frame

REPO = Path("/home/jovyan/scaleworm-student-lab")
MODEL = REPO / "99_runs/scaleworm_v8_tiled/weights/best.pt"
CONF, SLICE, OVERLAP = 0.40, 640, 0.2
TARGET = int(os.environ.get("TARGET", "42"))
OUT = REPO / "validation" / "subsample_2015_2016_stratified"
SCORES = OUT / "detector_scores.csv"
SCRATCH = Path("/tmp/claude-1000/-home-jovyan/d15fee80-de61-4938-b37b-519c259912ed/scratchpad/sub1516")


def pool() -> list[dict]:
    """Scene-1 frames with scene1_time_s + video_path, from the two scene sorts."""
    out = []
    for name in ("full_2015", "subsample_2016"):
        d = REPO / "scene_sorting" / name
        vids = {r["stem"]: r.get("video_path", "") for r in csv.DictReader((d / "manifest.csv").open())}
        for r in csv.DictReader((d / "sort_log.csv").open()):
            if r["decision"] == "scene1" and r.get("scene1_time_s"):
                out.append({"stem": r["stem"], "t": float(r["scene1_time_s"]),
                            "video_path": vids.get(r["stem"], ""), "src": name})
    return out


def score_all(frames: list[dict]) -> dict[str, int]:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    SCRATCH.mkdir(parents=True, exist_ok=True)
    done = {}
    if SCORES.exists():
        done = {r["stem"]: int(r["det_count"]) for r in csv.DictReader(SCORES.open())}
    todo = [f for f in frames if f["stem"] not in done and f["video_path"]]
    if todo:
        det = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(MODEL),
            confidence_threshold=CONF, device="cuda:0")
        OUT.mkdir(parents=True, exist_ok=True)
        new = not SCORES.exists()
        f = SCORES.open("a", newline="")
        w = csv.writer(f)
        if new:
            w.writerow(["stem", "det_count"])
        for i, fr in enumerate(todo, 1):
            png = SCRATCH / f"{fr['stem']}.png"
            if not extract_frame(Path(fr["video_path"]), fr["t"], png):
                continue
            r = get_sliced_prediction(str(png), det, slice_height=SLICE, slice_width=SLICE,
                                      overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP, verbose=0)
            png.unlink(missing_ok=True)
            done[fr["stem"]] = len(r.object_prediction_list)
            w.writerow([fr["stem"], len(r.object_prediction_list)])
            if i % 20 == 0:
                f.flush()
                print(f"  scored {i}/{len(todo)}", flush=True)
        f.close()
    return done


def stratified_pick(frames, scores) -> list[dict]:
    """Pick TARGET frames spanning detector-density terciles x time."""
    scored = [dict(f, c=scores[f["stem"]]) for f in frames if f["stem"] in scores]
    scored.sort(key=lambda x: x["c"])
    n = len(scored)
    terciles = [scored[: n // 3], scored[n // 3: 2 * n // 3], scored[2 * n // 3:]]
    per = max(1, TARGET // 3)
    picks = []
    for t in terciles:
        t.sort(key=lambda x: x["stem"])  # chronological within tercile
        step = max(1, len(t) / per)
        picks += [t[min(int(i * step), len(t) - 1)] for i in range(per)]
    # dedupe preserving order
    seen, out = set(), []
    for p in picks:
        if p["stem"] not in seen:
            seen.add(p["stem"])
            out.append(p)
    return out


def main() -> None:
    frames = pool()
    print(f"Scene-1 pool: {len(frames)} frames (2015 + 2016)")
    scores = score_all(frames)
    print(f"scored: {len(scores)} frames  (det count min/med/max: "
          f"{min(scores.values())}/{sorted(scores.values())[len(scores) // 2]}/{max(scores.values())})")
    picks = stratified_pick(frames, scores)

    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    for p in (OUT / "frames").glob("*.png"):
        p.unlink()
    rows = []
    for p in picks:
        png = OUT / "frames" / f"{p['stem']}.png"
        if extract_frame(Path(p["video_path"]), p["t"], png):
            rows.append({"frame_id": p["stem"], "year": p["stem"].split("-")[1][:4],
                         "scene1_time_s": p["t"], "detector_count": scores[p["stem"]],
                         "worm_count": "", "percent_cover": "", "counter": "",
                         "metric_note": "count OR %-cover per advisor decision", "video_path": p["video_path"]})
    with (OUT / "sort_sheet.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "README.md").write_text(
        "*AI-generated draft (Claude, Anthropic) — for review.*\n\n"
        "# 2015-2016 density-stratified hand-count subsample\n\n"
        f"{len(rows)} Scene-1 frames from the post-eruption era, picked to span detector-density terciles "
        "x time (sparse/moderate/dense, across 2015-2016). The detector is OUT-OF-DISTRIBUTION here "
        "(~28% recall) so `detector_count` is only a rough ordering used to stratify, NOT truth.\n\n"
        "PURPOSE: (1) check whether the automated detector tracks anything in this era, (2) anchor the "
        "true level, (3) decide the measurement metric. Fill EITHER `worm_count` (individual count) OR "
        "`percent_cover` per the advisor's count-vs-cover decision (2015-2016 is blocked on this). "
        "Frames are gitignored (regenerable from this script).\n")
    print(f"wrote {OUT}/sort_sheet.csv ({len(rows)} frames) + README")
    from collections import Counter
    print("picked by year:", dict(Counter(r["year"] for r in rows)),
          "| det-count range:", min(r["detector_count"] for r in rows), "-",
          max(r["detector_count"] for r in rows))


if __name__ == "__main__":
    main()
