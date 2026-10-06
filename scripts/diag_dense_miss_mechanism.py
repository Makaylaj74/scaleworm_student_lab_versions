"""Diagnose WHY the detector misses worms on dense frames: small-object size or crowding/occlusion?

For dense frames with human BOX ground truth, run the production detector and label every GT
worm found/missed (greedy centre-in-box, same metric as the gates). Then, comparing found vs
missed worms *within each frame* (so frame-level confounds -- blur, lighting, zoom -- cancel),
test which per-worm property predicts a miss:

  * size      -- normalized box area (small-object hypothesis -> tiling/SAHI could help)
  * crowding  -- nearest-neighbour distance + neighbours within r (occlusion hypothesis)
  * overlap   -- max IoU with any other GT box (direct occlusion -> tiling CANNOT help)

Decisive output: a within-frame-standardized logistic regression (found ~ z(size)+z(1/crowding)
+z(overlap)); the largest-magnitude standardized coefficient is the dominant miss driver.
Also per-feature found-vs-missed medians with bootstrap 95% CIs and Cohen's d.

Two strata scored independently (clear isolates density from blur; blurry is the harder regime):
  STRATUM=clear   dense held-out Monday clear frames (not in v5 train)
  STRATUM=blurry  dense Tuesday frames (all held out from v5)

Usage (thesis venv, ultralytics 8.4.62):
  STRATUM=clear  /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python scripts/diag_dense_miss_mechanism.py
  STRATUM=blurry /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python scripts/diag_dense_miss_mechanism.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from ultralytics import YOLO

REPO = Path("/home/jovyan/scaleworm-student-lab")
MODEL = os.environ.get("MODEL", str(REPO / "99_runs/scaleworm_v5_all/weights/best.pt"))
CONF = float(os.environ.get("CONF", "0.40"))
IMGSZ = int(os.environ.get("IMGSZ", "1280"))
STRATUM = os.environ.get("STRATUM", "clear")
RADIUS = 0.05  # normalized-coord neighbourhood radius for the crowding count
RNG = np.random.default_rng(20261006)


def frame_set() -> list[tuple[Path, Path]]:
    """(image, label) pairs for the chosen dense stratum, with box GT, held out from v5 train."""
    v5 = REPO / "datasets/scaleworm_retrain_all"
    train = {p.stem for p in (v5 / "images/train").glob("*")}
    if STRATUM == "clear":
        src = REPO / "validation/monday_manual_series"
        thr, clear_only = 18, True
    else:
        src = REPO / "validation/tuesday_manual_series"
        thr, clear_only = 25, False
    pairs = []
    for lp in sorted((src / "labels").glob("*.txt")):
        stem = lp.stem
        if stem in train:  # keep the "found" definition honest
            continue
        day = stem.split("-")[1][:8]
        if clear_only and not day < "20230811":
            continue
        boxes = [ln for ln in lp.read_text().splitlines() if ln.strip()]
        if len(boxes) < thr:
            continue
        img = next((src / "images").glob(f"{stem}.*"), None)
        if img:
            pairs.append((img, lp))
    return pairs


def iou(a, b):
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def found_flags(gt, preds):
    """gt: list (cx,cy,w,h) norm; preds: list (cx,cy,w,h,conf) norm. Greedy centre-in-box."""
    claimed = [False] * len(gt)
    for p in sorted(preds, key=lambda z: -z[4]):
        px1, py1 = p[0] - p[2] / 2, p[1] - p[3] / 2
        px2, py2 = p[0] + p[2] / 2, p[1] + p[3] / 2
        for j, g in enumerate(gt):
            if not claimed[j] and px1 <= g[0] <= px2 and py1 <= g[1] <= py2:
                claimed[j] = True
                break
    return claimed


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def main() -> None:
    model = YOLO(MODEL)
    pairs = frame_set()
    rows = []  # per-GT-worm: frame_idx, found, area, nn_dist, n_within, max_iou
    for fi, (img, lp) in enumerate(pairs):
        gt = []
        for ln in lp.read_text().splitlines():
            if ln.strip():
                _, cx, cy, w, h = ln.split()
                gt.append((float(cx), float(cy), float(w), float(h)))
        if len(gt) < 3:
            continue
        res = model(str(img), conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        H, W = res.orig_shape
        preds = []
        for b, c in zip(res.boxes.xywh.tolist(), res.boxes.conf.tolist()):
            preds.append((b[0] / W, b[1] / H, b[2] / W, b[3] / H, c))
        flags = found_flags(gt, preds)
        cen = np.array([[g[0], g[1]] for g in gt])
        for j, g in enumerate(gt):
            d = np.sqrt(((cen - cen[j]) ** 2).sum(1))
            d[j] = np.inf
            nn = float(d.min())
            n_within = int((d <= RADIUS).sum())
            max_iou = max((iou(g, gt[k]) for k in range(len(gt)) if k != j), default=0.0)
            rows.append([fi, int(flags[j]), g[2] * g[3], nn, n_within, max_iou])

    arr = np.array(rows, float)
    found = arr[:, 1] == 1
    area, nn, nwin, miou = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
    n_tot = len(arr)
    n_found = int(found.sum())

    print(f"\n{'=' * 66}\nDENSE MISS MECHANISM  ({STRATUM})  model={Path(MODEL).stem} conf={CONF}")
    print(f"frames={len(set(arr[:, 0]))}  GT worms={n_tot}  "
          f"found={n_found} ({n_found / n_tot:.0%})  missed={n_tot - n_found}\n")

    def line(name, vals):
        f_, m_ = vals[found], vals[~found]
        flo, fhi = boot_ci(f_)
        mlo, mhi = boot_ci(m_)
        print(f"  {name:16s} found med={np.median(f_):.4f} [{flo:.4f},{fhi:.4f}]   "
              f"missed med={np.median(m_):.4f} [{mlo:.4f},{mhi:.4f}]   d={cohens_d(m_, f_):+.2f}")

    print("found vs missed (per-worm):")
    line("box area", area)
    line("nn distance", nn)      # missed > found would mean missed are MORE isolated (not crowding)
    line("neighbours<=r", nwin)  # missed > found => missed are MORE crowded
    line("max IoU(overlap)", miou)

    # within-frame standardized logistic regression (frame effects removed)
    feats = np.column_stack([area, -nn, nwin, miou])  # -nn so higher = more crowded
    names = ["size(area)", "crowding(-nn)", "crowding(nwin)", "overlap(maxIoU)"]
    z = np.full_like(feats, np.nan)
    for fi in np.unique(arr[:, 0]):
        m = arr[:, 0] == fi
        sub = feats[m]
        sd = sub.std(0)
        sd[sd == 0] = 1.0
        z[m] = (sub - sub.mean(0)) / sd
    keep = ~np.isnan(z).any(1)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(z[keep], found[keep].astype(int))
    print("\nwithin-frame standardized logistic reg  (predicting FOUND; "
          "negative coef => drives a MISS):")
    for nm, co in zip(names, clf.coef_[0]):
        tag = " <== dominant" if abs(co) == max(abs(clf.coef_[0])) else ""
        print(f"  {nm:18s} coef={co:+.3f}{tag}")
    print("\nread: if overlap/crowding dominates (large negative) -> OCCLUSION, tiling won't fix.")
    print("      if size dominates (large positive) -> small-object, tile-training could help.")


if __name__ == "__main__":
    main()
