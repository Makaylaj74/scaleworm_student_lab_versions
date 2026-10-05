"""Two-tail SAHI test + tuning sweep: can tiled inference fix the density under-count?

The flat ÷0.72 recall factor and higher-res inference both failed because the bias depends on
TRUE density: dense peaks stay under-counted (~61% @>22 worms) while sparse frames are fine.
SAHI (Slicing-Aided Hyper Inference) runs the detector on overlapping tiles at native
resolution then merges — attacking crowding directly rather than correcting after the fact.

First pass (slice 640, GREEDYNMM/IOS 0.5) gave a BORDERLINE result: dense count-ratio
61%->70% but sparse inflated 100%->111% with precision drops (tile-edge false positives). The
FP inflation is a merge/tile-size artifact, so this script sweeps SMALLER tiles + TIGHTER
cross-tile merge to see if dense clears ~80% WITHOUT inflating sparse.

Test on the two tails, LEAKAGE-FREE val frames only (v5_all never trained on the val split):
  - DENSE  (manual GT > 22 worms): does SAHI count-ratio climb toward 100%?
  - SPARSE (manual GT <= 6 worms): does SAHI AVOID inflating the count (tile FPs)?

Plain v5_all vs SAHI-wrapped v5_all on identical frames (same model, so the delta is a clean
read on tiling). Point-matched recall/precision + count-ratio vs MJ box centres, conf 0.40.

Usage (thesis venv, ultralytics 8.4.62 + sahi):
  /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/test_sahi_two_tail.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from gate_retrain import point_match
from gate_v6_blurry import box_centers

REPO = Path(__file__).resolve().parent.parent
MS = REPO / "validation/monday_manual_series"
MODEL = REPO / "99_runs/scaleworm_v5_all/weights/best.pt"
OUT = REPO / "validation/sahi_two_tail.csv"

CONF = 0.40
CLEAR_END = "2023-08-11"
DENSE_GT, SPARSE_GT = 22, 6

# SAHI configs to sweep: (label, slice_px, overlap, postprocess_type, match_metric, match_thr)
# Tighter merge = lower match threshold (merges near-duplicate cross-tile boxes -> fewer FPs).
SAHI_CONFIGS = [
    ("sahi-640/0.5", 640, 0.2, "GREEDYNMM", "IOS", 0.5),  # first-pass baseline
    ("sahi-512/0.4", 512, 0.2, "GREEDYNMM", "IOS", 0.4),  # smaller tiles, tighter
    ("sahi-512/0.3", 512, 0.2, "GREEDYNMM", "IOS", 0.3),  # smaller tiles, tighter still
    ("sahi-384/0.3", 384, 0.2, "GREEDYNMM", "IOS", 0.3),  # smallest tiles, tighter
]


def parse_count(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def select_frames() -> dict[str, list[str]]:
    import pandas as pd

    m = pd.read_csv(MS / "frame_manifest.csv")
    m["dt"] = pd.to_datetime(m["datetime_utc"], errors="coerce").dt.tz_localize(None)
    m["gtc"] = m["worm_count"].map(parse_count)
    c = m[(m["dt"] < CLEAR_END) & m["gtc"].notna() & (m["split"] == "val")]
    dense = sorted(c[c["gtc"] > DENSE_GT]["frame_id"])
    sparse = sorted(c[c["gtc"] <= SPARSE_GT]["frame_id"])
    return {"dense": dense, "sparse": sparse}


def plain_boxes(model, png: Path):
    res = model(str(png), conf=CONF, verbose=False)[0]
    return [
        (float(a), float(b), float(c), float(d), float(s))
        for (a, b, c, d), s in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())
    ]


def sahi_boxes(det, png: Path, cfg):
    from sahi.predict import get_sliced_prediction

    _, slc, ov, ptype, metric, thr = cfg
    r = get_sliced_prediction(
        str(png),
        det,
        slice_height=slc,
        slice_width=slc,
        overlap_height_ratio=ov,
        overlap_width_ratio=ov,
        postprocess_type=ptype,
        postprocess_match_metric=metric,
        postprocess_match_threshold=thr,
        verbose=0,
    )
    return [
        (p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy, p.score.value)
        for p in r.object_prediction_list
    ]


def score(tail, frames, pred_fn, label):
    import cv2

    recs = []
    for fid in frames:
        png, lbl = MS / "images" / f"{fid}.png", MS / "labels" / f"{fid}.txt"
        if not png.exists() or not lbl.exists():
            continue
        preds = pred_fn(png)
        H, W = cv2.imread(str(png)).shape[:2]
        pts = box_centers(lbl, W, H)
        tp, fp, fn = point_match(preds, pts)
        recs.append({"gt": len(pts), "pred": len(preds), "tp": tp, "fp": fp, "fn": fn})
    gt = sum(r["gt"] for r in recs)
    pred = sum(r["pred"] for r in recs)
    tp = sum(r["tp"] for r in recs)
    fp = sum(r["fp"] for r in recs)
    fn = sum(r["fn"] for r in recs)
    cr = pred / gt if gt else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    print(
        f"  {label:16s} [{tail:6s}] n={len(recs):2d} human={gt:4d} model={pred:4d}  "
        f"count={cr:6.1%}  recall={rec:6.1%}  prec={prec:6.1%}"
    )
    return {
        "method": label,
        "tail": tail,
        "n": len(recs),
        "human": gt,
        "model": pred,
        "count_ratio": round(cr, 4),
        "point_recall": round(rec, 4),
        "precision": round(prec, 4),
    }


def main() -> None:
    from sahi import AutoDetectionModel
    from ultralytics import YOLO

    sel = select_frames()
    print(
        f"two-tail val frames: dense={len(sel['dense'])} (GT>{DENSE_GT})  "
        f"sparse={len(sel['sparse'])} (GT<={SPARSE_GT})  conf={CONF}\n"
    )

    plain = YOLO(str(MODEL))
    try:
        det = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(MODEL),
            confidence_threshold=CONF,
            device="cpu",
        )
    except Exception:  # noqa: BLE001  (model_type fallback: any load error -> try yolov8)
        det = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=str(MODEL),
            confidence_threshold=CONF,
            device="cpu",
        )

    rows = []
    print("--- plain baseline ---")
    for tail in ("dense", "sparse"):
        rows.append(
            score(tail, sel[tail], lambda p: plain_boxes(plain, p), "plain v5_all")
        )
    for cfg in SAHI_CONFIGS:
        print(f"--- {cfg[0]} (slice {cfg[1]}, {cfg[3]}/{cfg[4]} thr {cfg[5]}) ---")
        for tail in ("dense", "sparse"):
            rows.append(
                score(tail, sel[tail], lambda p, c=cfg: sahi_boxes(det, p, c), cfg[0])
            )

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "tail",
                "method",
                "n",
                "human",
                "model",
                "count_ratio",
                "point_recall",
                "precision",
            ],
        )
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["tail"], r["method"])))
    print(f"\nwrote {OUT}")

    # verdict: pick the SAHI config that best lifts dense while keeping sparse near 100%
    def get(method, tail):
        return next(r for r in rows if r["method"] == method and r["tail"] == tail)

    print(
        "\n=== SUMMARY (dense count-ratio toward 100% is the goal; sparse should stay ~100%) ==="
    )
    print(
        f"  {'config':16s} {'dense count':>12s} {'dense recall':>13s} {'sparse count':>13s} {'sparse prec':>12s}"
    )
    best, best_score = None, -1.0
    for method in ["plain v5_all"] + [c[0] for c in SAHI_CONFIGS]:
        d, s = get(method, "dense"), get(method, "sparse")
        print(
            f"  {method:16s} {d['count_ratio']:>12.0%} {d['point_recall']:>13.0%} "
            f"{s['count_ratio']:>13.0%} {s['precision']:>12.0%}"
        )
        if method != "plain v5_all":
            # objective: maximise dense count-ratio, penalise sparse over-count beyond 100%
            obj = d["count_ratio"] - max(0.0, s["count_ratio"] - 1.0)
            if obj > best_score:
                best, best_score = method, obj
    bd, bs = get(best, "dense"), get("plain v5_all", "dense")
    print("\n=== VERDICT ===")
    print(f"  best SAHI config: {best}")
    print(
        f"  dense count-ratio: plain {bs['count_ratio']:.0%} -> {bd['count_ratio']:.0%}"
    )
    if bd["count_ratio"] >= 0.80 and get(best, "sparse")["count_ratio"] <= 1.15:
        print(
            "  => OPTION 2 VIABLE: dense clears ~80% without sparse inflation. Use SAHI for dense."
        )
    elif bd["count_ratio"] > bs["count_ratio"] + 0.05:
        print(
            "  => PARTIAL: tiling helps but dense still short of 80% / some sparse cost. Ceiling likely."
        )
    else:
        print(
            "  => CEILING CONFIRMED: tuning did not clear the dense wall. Cap claim at conservative index."
        )


if __name__ == "__main__":
    main()
