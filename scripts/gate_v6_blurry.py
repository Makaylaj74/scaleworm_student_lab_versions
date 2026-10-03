"""Acceptance gate for v6_blurry — the HONEST blurry test v5 never had.

v5's blurry side was gated on the under-counted Monday blurry frames
(`model_comparison_handcount.csv`) — the same bad labels it trained on, so its
"94% blurry" was circular and was retracted. v6_blurry retrains the blurry side on
MJ's properly box-corrected **dense Tuesday** frames (mean ~25/frame) and holds out a
day-disjoint portion as the gate (`datasets/scaleworm_v6_blurry/tuesday_blurry_test.txt`).

This scores v0 (mushroom.pt baseline), v5_all, and v6_blurry on the SAME held-out
frames so the comparison is apples-to-apples:
  - CLEAR  : validation/clear_window_handcount (39 frames) — independent CLICK ground
             truth, conf 0.40 (the clear operating point). Confirms v6 did not regress.
  - BLURRY : the 23 held-out dense Tuesday frames — MJ box centres as ground truth,
             conf 0.25 (the blurry operating point). The headline number.

Metrics per (model, era): point-matched recall / precision / F1 (greedy point-in-box,
the honest detector metric) + count-ratio recall, count MAE / bias, with per-frame
percentile-bootstrap 95% CIs. Leakage is re-verified against v6_blurry's actual
train/val stems.

Usage (thesis venv, ultralytics 8.4.62, for comparability with the v5 baselines):
  /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/gate_v6_blurry.py
"""

from __future__ import annotations

import csv
import glob
import json
import statistics
from pathlib import Path

from gate_retrain import boot_ci, day_of, point_match  # reuse the tested helpers

REPO = Path("/home/jovyan/scaleworm-student-lab")
CLEAR = REPO / "validation" / "clear_window_handcount"
TUE = REPO / "validation" / "tuesday_manual_series"
V6_DS = REPO / "datasets" / "scaleworm_v6_blurry"
TEST_LIST = V6_DS / "tuesday_blurry_test.txt"
OUT = REPO / "validation" / "v6_blurry_gate.csv"

MODELS = {
    "v0_mushroom": REPO / "mushroom.pt",
    "v5_all": REPO / "99_runs" / "scaleworm_v5_all" / "weights" / "best.pt",
    "v6_blurry": REPO / "99_runs" / "scaleworm_v6_blurry" / "weights" / "best.pt",
}
CONF = {"clear": 0.40, "blurry": 0.25}  # per-regime operating points (v5 note)


def v6_training_stems() -> set[str]:
    stems = set()
    for split in ("train", "val"):
        for p in glob.glob(str(V6_DS / "labels" / split / "*.txt")):
            stems.add(Path(p).stem)
    return stems


def box_centers(label_path: Path, W: int, H: int) -> list[tuple[float, float]]:
    """YOLO box centres (px) from a label file."""
    pts = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _, cx, cy, _, _ = line.split()
        pts.append((float(cx) * W, float(cy) * H))
    return pts


def predict(model, png: Path, conf: float):
    res = model(str(png), conf=conf, verbose=False)[0]
    H, W = res.orig_shape
    preds = [
        (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
        for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())
    ]
    return preds, W, H


def report(label: str, era: str, recs: list[dict]) -> dict:
    n = len(recs)
    TP = sum(r["tp"] for r in recs)
    FP = sum(r["fp"] for r in recs)
    FN = sum(r["fn"] for r in recs)
    model_tot = sum(r["pred"] for r in recs)
    human_tot = sum(r["gt"] for r in recs)
    errs = [r["pred"] - r["gt"] for r in recs]
    count_recall = model_tot / human_tot if human_tot else 0.0
    det_recall = TP / (TP + FN) if TP + FN else float("nan")
    prec = TP / (TP + FP) if TP + FP else float("nan")
    f1 = 2 * prec * det_recall / (prec + det_recall) if (prec + det_recall) else float("nan")
    rlo, rhi = boot_ci([r["tp"] / r["gt"] for r in recs if r["gt"]])
    clo, chi = boot_ci([r["pred"] / r["gt"] for r in recs if r["gt"]])
    print(f"\n=== {label} · {era}  (n={n}) ===")
    print(f"  human total/mean   : {human_tot} / {human_tot / n:.2f} per frame")
    print(f"  model total/mean   : {model_tot} / {model_tot / n:.2f} per frame")
    print(f"  COUNT-ratio recall : {count_recall:.1%}  (per-frame 95% CI [{clo:.1%},{chi:.1%}])")
    print(f"  POINT recall       : {det_recall:.1%}  (TP={TP} FN={FN}, 95% CI [{rlo:.1%},{rhi:.1%}])")
    print(f"  POINT precision/F1 : {prec:.1%} (FP={FP}) / {f1:.1%}")
    print(f"  count MAE / bias   : {statistics.mean(abs(e) for e in errs):.2f} / {statistics.mean(errs):+.2f}")
    return {
        "model": label,
        "era": era,
        "conf": CONF.get(era.split()[0].lower(), ""),
        "n": n,
        "human": human_tot,
        "model_total": model_tot,
        "count_recall": round(count_recall, 4),
        "point_recall": round(det_recall, 4),
        "point_recall_ci_lo": round(rlo, 4),
        "point_recall_ci_hi": round(rhi, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "mae": round(statistics.mean(abs(e) for e in errs), 3),
        "bias": round(statistics.mean(errs), 3),
    }


def clear_recs(model):
    rows = [
        r
        for r in csv.DictReader((CLEAR / "handcount_sheet.csv").open())
        if r.get("frame_ready") == "Y" and (r.get("worm_count") or "").strip() != ""
    ]
    recs = []
    for r in rows:
        fid = r["frame_id"]
        png, clk = CLEAR / "frames" / f"{fid}.png", CLEAR / "clicks" / f"{fid}.json"
        if not png.exists() or not clk.exists():
            continue
        pts = [tuple(p) for p in json.load(clk.open())["clicks"]]
        preds, _W, _H = predict(model, png, CONF["clear"])
        tp, fp, fn = point_match(preds, pts)
        recs.append({"gt": len(pts), "pred": len(preds), "tp": tp, "fp": fp, "fn": fn})
    return recs


def blurry_recs(model):
    test = [s.strip() for s in TEST_LIST.read_text().splitlines() if s.strip()]
    recs = []
    for fid in test:
        png, lbl = TUE / "images" / f"{fid}.png", TUE / "labels" / f"{fid}.txt"
        if not png.exists() or not lbl.exists():
            continue
        preds, W, H = predict(model, png, CONF["blurry"])
        pts = box_centers(lbl, W, H)
        tp, fp, fn = point_match(preds, pts)
        recs.append({"gt": len(pts), "pred": len(preds), "tp": tp, "fp": fp, "fn": fn})
    return recs


def main() -> None:
    from ultralytics import YOLO

    # --- leakage re-check against v6_blurry's actual training stems ---
    tstems = v6_training_stems()
    tdays = {day_of(s) for s in tstems}
    test = [s.strip() for s in TEST_LIST.read_text().splitlines() if s.strip()]
    clear_ids = [
        r["frame_id"]
        for r in csv.DictReader((CLEAR / "handcount_sheet.csv").open())
        if r.get("frame_ready") == "Y" and (r.get("worm_count") or "").strip() != ""
    ]
    leaked = [g for g in (test + clear_ids) if day_of(g) in tdays or g in tstems]
    assert not leaked, f"LEAKAGE: gate frames in v6 training: {leaked}"
    print(
        f"leakage OK: 0/{len(test) + len(clear_ids)} gate frames in v6_blurry train/val "
        f"({len(tstems)} train stems, {len(tdays)} train days)"
    )

    summ = []
    blurry_by_model: dict[str, list[dict]] = {}
    for label, path in MODELS.items():
        model = YOLO(str(path))
        summ.append(report(label, "CLEAR (clicks GT)", clear_recs(model)))
        brecs = blurry_recs(model)
        blurry_by_model[label] = brecs
        summ.append(report(label, "BLURRY (Tuesday-dense, MJ-box GT)", brecs))

    # --- paired v6-vs-v5 test on the SAME blurry frames (defensible, not unpaired CIs) ---
    v5, v6 = blurry_by_model["v5_all"], blurry_by_model["v6_blurry"]
    # per-frame point recall (TP/gt); same frame order from TEST_LIST
    d_recall = [
        (b["tp"] / b["gt"]) - (a["tp"] / a["gt"])
        for a, b in zip(v5, v6)
        if a["gt"] and b["gt"]
    ]
    mean_d = statistics.mean(d_recall)
    lo, hi = boot_ci(d_recall)
    wins = sum(1 for d in d_recall if d > 0)
    print("\n=== PAIRED  v6_blurry - v5_all  ·  blurry (per-frame point recall) ===")
    print(f"  mean diff: {mean_d:+.1%}   95% CI [{lo:+.1%}, {hi:+.1%}]   "
          f"(v6 higher on {wins}/{len(d_recall)} frames)")
    verdict = "NO significant improvement (CI spans 0)" if lo <= 0 <= hi else (
        "v6 significantly better" if lo > 0 else "v6 significantly WORSE")
    print(f"  verdict: {verdict}")

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
