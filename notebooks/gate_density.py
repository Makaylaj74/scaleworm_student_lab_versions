"""Gate the CSRNet density head on held-out val frames.

count_density = integral of the predicted density map. Compared to:
  - manual count (truth = # boxes in the original single-class label)
  - v5_all baseline: predicted count / 0.72 (the current production index)
broken out by density bin -- the dense regime (manual >= 20) is where the
box detector saturates and is the entire reason for trying density regression.
Same val frames and same manual-truth definition as the 2-class gate, so the
numbers are directly comparable to the three previously-closed approaches.
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import density_head as D

OUT = D.OUT
# v5_all baseline preds (manual + box count) vendored alongside the run so the
# gate is reproducible in-repo; originally produced by the coverage-index run.
V5_PREDS = OUT / "coverage_index_preds.csv"
# Okabe-Ito
C_DENS, C_V5 = "#0072B2", "#D55E00"


def manual_count(stem: str) -> int:
    fp = D.DS / "labels/val" / f"{stem}.txt"
    return sum(1 for ln in fp.read_text().splitlines() if ln.strip())


def r2(pred, truth):
    b, a = np.polyfit(pred, truth, 1)
    return 1 - np.sum((truth - (a + b * pred)) ** 2) / np.sum(
        (truth - truth.mean()) ** 2
    )


def main() -> None:
    device = "cuda"
    model = D.CSRNet().to(device)
    model.load_state_dict(torch.load(OUT / "best.pt", map_location=device))
    model.eval()
    norm = transforms.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)

    v5 = {}
    for row in csv.DictReader(V5_PREDS.open()):
        if row["split"] == "val":
            v5[row["stem"]] = float(row["pred_count"])

    rows = []
    with torch.no_grad():
        for img_fp in sorted((D.DS / "images/val").glob("*.png")):
            stem = img_fp.stem
            img = (
                Image.open(img_fp)
                .convert("RGB")
                .resize((D.IN_W, D.IN_H), Image.BILINEAR)
            )
            x = norm(transforms.functional.to_tensor(img))[None].to(device)
            dens = float(model(x).sum().item())
            rows.append((stem, manual_count(stem), dens, v5.get(stem, np.nan) / 0.72))

    arr = np.array([(m, d, v) for _, m, d, v in rows], float)
    m, d, v = arr[:, 0], arr[:, 1], arr[:, 2]

    print(f"density-head gate  (val n={len(m)})\n")
    print(f"{'predictor':<20} {'r':>6} {'R2':>6} {'MAE':>6} {'overall ratio':>14}")
    for name, p in [("density integral", d), ("v5 count/0.72", v)]:
        ok = ~np.isnan(p)
        rr = np.corrcoef(p[ok], m[ok])[0, 1]
        mae = np.abs(p[ok] - m[ok]).mean()
        print(
            f"{name:<20} {rr:>6.3f} {r2(p[ok], m[ok]):>6.3f} {mae:>6.2f} "
            f"{(p[ok].sum() / m[ok].sum()):>13.2f}x"
        )

    print("\ndense-regime ratio to manual (1.00 = perfect):")
    bins = [(0, 10), (10, 20), (20, 30), (30, 999)]
    brows = []
    for lo, hi in bins:
        mb = (m >= lo) & (m < hi)
        if mb.sum() < 1:
            brows.append((f"{lo}-{hi}", 0, np.nan, np.nan))
            continue
        dr = (d[mb] / m[mb]).mean()
        vok = mb & ~np.isnan(v)
        vr = (v[vok] / m[vok]).mean() if vok.sum() else np.nan
        brows.append((f"{lo}-{hi}", int(mb.sum()), dr, vr))
        print(
            f"  manual {lo:2d}-{hi:<3d} (n={mb.sum():2d}): "
            f"density {dr:.2f}x   v5/0.72 {vr:.2f}x"
        )

    # figure: scatter + per-bin ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    hi = max(m.max(), d.max(), np.nanmax(v)) * 1.05
    ax1.plot([0, hi], [0, hi], "k--", lw=1, label="1:1")
    ax1.scatter(m, d, s=28, c=C_DENS, alpha=0.7, label="density integral")
    ax1.scatter(m, v, s=28, c=C_V5, marker="^", alpha=0.7, label="v5 count/0.72")
    ax1.set_xlabel("manual count (truth)")
    ax1.set_ylabel("predicted count")
    ax1.set_title("Predicted vs manual (val)")
    ax1.legend(frameon=False)

    labels = [b[0] for b in brows]
    xs = np.arange(len(labels))
    ax2.axhline(1.0, color="k", lw=1, ls="--")
    ax2.bar(xs - 0.2, [b[2] for b in brows], 0.4, color=C_DENS, label="density")
    ax2.bar(xs + 0.2, [b[3] for b in brows], 0.4, color=C_V5, label="v5/0.72")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{l}\n(n={b[1]})" for l, b in zip(labels, brows)])
    ax2.set_xlabel("manual-count bin")
    ax2.set_ylabel("mean predicted / manual")
    ax2.set_title("Ratio to truth by density bin")
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "gate_density.png", dpi=150)
    print(f"\nfigure -> {OUT / 'gate_density.png'}")

    with (OUT / "gate_density_results.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["stem", "manual", "density_count", "v5_corr"])
        wr.writerows(rows)
    print(f"csv    -> {OUT / 'gate_density_results.csv'}")


if __name__ == "__main__":
    main()
