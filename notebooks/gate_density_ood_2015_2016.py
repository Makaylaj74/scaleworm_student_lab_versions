"""OOD gate: CSRNet density head on 2015-2016 frames (the actual motivation).

The density head was trained ONLY on 2021-2024 Monday frames (tower scene). The
2015-2016 record is a visually different early-post-eruption mound/tubeworm
scene AND dense (the Axial 2015 eruption recovery is the whole reason for the
project). This is therefore a true out-of-distribution test, and -- unlike the
held-out val -- it has hand-count ground truth (MJ, 2026-10-07).

Baseline = the box detector count the anchor sheet was scored with
(detector_count), /0.72 like the production index. Compared against MJ's
worm_count. n=35 frames with a filled count (33x 2015, 2x 2016).

Caveats baked into the read: single-rater GT (~2x IRR uncertainty established
for this task); n=35, dense-skewed (mean ~25); genuinely OOD scene.

Convention for CIs: non-parametric percentile bootstrap over frames,
n_boot=10000, seed=0, 95% interval.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import density_head as D

ANCHOR = Path(
    "/home/jovyan/scaleworm-student-lab/validation/subsample_2015_2016_stratified"
)
SHEET = ANCHOR / "handcount_sheet.csv"
FRAMES = ANCHOR / "frames"
N_BOOT = 10000
SEED = 0


def load_truth() -> dict[str, tuple[int, int]]:
    """frame_id -> (gt_worm_count, detector_count) for rows with a GT count."""
    out = {}
    for row in csv.DictReader(SHEET.open()):
        wc = (row["worm_count"] or "").strip()
        if wc:
            out[row["frame_id"]] = (int(wc), int(row["detector_count"]))
    return out


def ci(s: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def boot(fn, *arrs, seed=SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(arrs[0])
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        vals[i] = fn(*[a[idx] for a in arrs])
    return ci(vals)


def main() -> None:
    truth = load_truth()
    device = "cuda"
    model = D.CSRNet().to(device)
    model.load_state_dict(torch.load(D.OUT / "best.pt", map_location=device))
    model.eval()
    norm = transforms.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)

    stems, gt, det, dens = [], [], [], []
    with torch.no_grad():
        for stem, (g, dc) in sorted(truth.items()):
            fp = FRAMES / f"{stem}.png"
            if not fp.exists():
                continue
            img = Image.open(fp).convert("RGB").resize(
                (D.IN_W, D.IN_H), Image.BILINEAR)
            x = norm(transforms.functional.to_tensor(img))[None].to(device)
            stems.append(stem)
            gt.append(g)
            det.append(dc)
            dens.append(float(model(x).sum().item()))

    gt = np.array(gt, float)
    det = np.array(det, float)
    dens = np.array(dens, float)
    det_c = det / 0.72

    def mae(p, t):
        return np.abs(p - t).mean()

    def ratio(p, t):
        return p.sum() / t.sum()

    def pearson(p, t):
        return np.corrcoef(p, t)[0, 1]

    print(f"OOD gate 2015-2016  (n={len(gt)}, GT mean {gt.mean():.1f}/fr, "
          f"n_boot={N_BOOT} seed={SEED} 95% CI)\n")
    print(f"{'predictor':<20} {'r':>18} {'MAE':>18} {'overall ratio':>20}")
    for name, p in [("density integral", dens), ("detector/0.72", det_c)]:
        rlo, rhi = boot(pearson, p, gt)
        mlo, mhi = boot(mae, p, gt)
        olo, ohi = boot(ratio, p, gt)
        print(f"{name:<20} "
              f"{pearson(p, gt):.3f} ({rlo:.2f},{rhi:.2f})   "
              f"{mae(p, gt):5.2f} ({mlo:.1f},{mhi:.1f})   "
              f"{ratio(p, gt):.2f}x ({olo:.2f},{ohi:.2f})")

    print("\nper-frame (sorted by GT):")
    print(f"  {'frame':<32} {'GT':>4} {'density':>8} {'det/0.72':>9}")
    order = np.argsort(gt)
    for i in order:
        print(f"  {stems[i]:<32} {int(gt[i]):>4} {dens[i]:>8.1f} {det_c[i]:>9.1f}")


if __name__ == "__main__":
    main()
