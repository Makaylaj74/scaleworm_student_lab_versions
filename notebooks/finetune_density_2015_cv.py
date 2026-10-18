"""Fine-tune the CSRNet density head with the 2015-2016 hand-counts, 5-fold CV.

The density head trained on 2021-24 (tower) did NOT transfer to the 2015-16
mound/tubeworm scene (OOD gate: 0.34x, ~no better than the box detector). This
tests whether that wall is a fixable data-coverage gap: warm-start from the
2021-24 model and fine-tune including MJ's 2015-16 click annotations.

n=35 click-annotated 2015-16 frames is too small for a fixed holdout, so we use
5-fold CV stratified by density: each fold warm-starts from the committed
2021-24 best.pt and fine-tunes on (full 2021-24 clear train 421 frames UNION
the 4/5 in-fold 2015-16 frames, with the 2015-16 frames oversampled to balance),
then predicts the held-out 1/5. Aggregating the held-out predictions gives a
leakage-free OOD-after-fine-tune estimate over all 35 frames.

Ground truth for 2015-16 = click points (count == MJ worm_count exactly, 0
mismatches). Density maps are built from the click points directly (unit mass
each). The 2015-16 clicks have no box extents, so the Gaussian sigma is the
median per-worm sigma DERIVED from the 2021-24 boxes (same species/pipeline,
not a borrowed constant); the map integral equals the exact count regardless
of sigma, so count accuracy is sigma-invariant.

Also re-checks 2021-24 clear val retention (fold 0 model) so we see whether the
in-distribution win survives the fine-tune.

Config via env: EPOCHS (40), OVERSAMPLE (10), LR (1e-5), FOLDS (5), WORKERS (6).
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

import density_head as D

ANCHOR = Path(
    "/home/jovyan/scaleworm-student-lab/validation/subsample_2015_2016_stratified"
)
SHEET = ANCHOR / "handcount_sheet.csv"
FRAMES = ANCHOR / "frames"
CLICKS = ANCHOR / "clicks"
OUT = Path(__file__).resolve().parent / "csrnet_ft2015_run"
OUT.mkdir(exist_ok=True)

EPOCHS = int(os.environ.get("EPOCHS", "40"))
OVERSAMPLE = int(os.environ.get("OVERSAMPLE", "10"))
LR = float(os.environ.get("LR", "1e-5"))
FOLDS = int(os.environ.get("FOLDS", "5"))
WORKERS = min(8, int(os.environ.get("WORKERS", "6")))  # lab cap: <= 24
SEED = 0
N_BOOT = 10000


def derive_sigma_out() -> float:
    """Median per-worm output-grid sigma from the 2021-24 training boxes."""
    sigmas = []
    for fp in (D.DS / "labels/train").glob("*.txt"):
        for _cx, _cy, bw, bh in D.read_boxes(fp):
            bs_out = min(bw * D.OUT_W, bh * D.OUT_H)
            sigmas.append(max(1.5, 0.5 * bs_out))
    return float(np.median(sigmas)) if sigmas else 2.0


SIGMA_OUT = derive_sigma_out()


def load_clicks(stem: str) -> np.ndarray:
    """-> (N,2) normalized (x,y) in [0,1] from full-frame pixel clicks."""
    d = json.loads((CLICKS / f"{stem}.json").read_text())
    pts = np.array(d["clicks"], float) if d["clicks"] else np.zeros((0, 2))
    if len(pts):
        pts[:, 0] /= D.FULL_W
        pts[:, 1] /= D.FULL_H
    return pts


def density_map_points(pts: np.ndarray) -> np.ndarray:
    """(OUT_H, OUT_W) density map from normalized points; integral == len(pts)."""
    dm = np.zeros((D.OUT_H, D.OUT_W), np.float32)
    sigma = SIGMA_OUT
    rad = int(np.ceil(3 * sigma))
    for nx, ny in pts:
        ox, oy = nx * D.OUT_W, ny * D.OUT_H
        x0, x1 = max(0, int(ox) - rad), min(D.OUT_W, int(ox) + rad + 1)
        y0, y1 = max(0, int(oy) - rad), min(D.OUT_H, int(oy) + rad + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1) - ox
        ys = np.arange(y0, y1) - oy
        g = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2 * sigma**2))
        s = g.sum()
        if s > 0:
            dm[y0:y1, x0:x1] += (g / s).astype(np.float32)
    return dm


class Clicks2015(Dataset):
    def __init__(self, stems: list[str], train: bool):
        self.stems = stems
        self.train = train
        self.norm = transforms.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.stems)

    def _augment(self, img: Image.Image, pts: np.ndarray):
        if random.random() < 0.5 and len(pts):
            s = random.uniform(0.6, 1.0)
            x0 = random.uniform(0, 1 - s)
            y0 = random.uniform(0, 1 - s)
            img = img.crop(
                (int(x0 * D.FULL_W), int(y0 * D.FULL_H),
                 int((x0 + s) * D.FULL_W), int((y0 + s) * D.FULL_H))
            )
            keep = ((pts[:, 0] >= x0) & (pts[:, 0] < x0 + s)
                    & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + s))
            pts = pts[keep].copy()
            if len(pts):
                pts[:, 0] = (pts[:, 0] - x0) / s
                pts[:, 1] = (pts[:, 1] - y0) / s
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(pts):
                pts[:, 0] = 1.0 - pts[:, 0]
        return img, pts

    def __getitem__(self, i: int):
        stem = self.stems[i]
        img = Image.open(FRAMES / f"{stem}.png").convert("RGB")
        pts = load_clicks(stem)
        if self.train:
            img, pts = self._augment(img, pts)
        img = img.resize((D.IN_W, D.IN_H), Image.BILINEAR)
        x = self.norm(transforms.functional.to_tensor(img))
        dm = torch.from_numpy(density_map_points(pts))[None]
        return x, dm, float(len(pts))


def load_gt() -> list[tuple[str, int, int]]:
    """(stem, gt_count, detector_count) for rows with a filled worm_count."""
    out = []
    for row in csv.DictReader(SHEET.open()):
        wc = (row["worm_count"] or "").strip()
        if wc and (FRAMES / f"{row['frame_id']}.png").exists():
            out.append((row["frame_id"], int(wc), int(row["detector_count"])))
    return out


def make_folds(items: list[tuple[str, int, int]]) -> list[list[int]]:
    """Density-stratified k folds: rank by GT, round-robin assign."""
    order = sorted(range(len(items)), key=lambda i: items[i][1])
    folds = [[] for _ in range(FOLDS)]
    for rank, idx in enumerate(order):
        folds[rank % FOLDS].append(idx)
    return folds


@torch.no_grad()
def predict(model, stems: list[str], device) -> np.ndarray:
    model.eval()
    norm = transforms.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)
    out = []
    for stem in stems:
        img = Image.open(FRAMES / f"{stem}.png").convert("RGB").resize(
            (D.IN_W, D.IN_H), Image.BILINEAR)
        x = norm(transforms.functional.to_tensor(img))[None].to(device)
        out.append(float(model(x).sum().item()))
    return np.array(out)


def train_fold(train_stems: list[str], device, log) -> nn.Module:
    model = D.CSRNet().to(device)
    model.load_state_dict(torch.load(D.OUT / "best.pt", map_location=device))
    clear = D.WormDensity("train", True)
    worms = Clicks2015(train_stems, True)
    ds = ConcatDataset([clear] + [worms] * OVERSAMPLE)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=WORKERS,
                        drop_last=True, pin_memory=True)
    crit = nn.MSELoss(reduction="sum")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for x, dm, _ in loader:
            x, dm = x.to(device), dm.to(device)
            opt.zero_grad()
            loss = crit(model(x), dm) / x.size(0)
            loss.backward()
            opt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == EPOCHS - 1:
            msg = f"    ep {ep:3d}  loss {tot / len(loader):8.4f}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
    return model


def ci(s: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def boot(fn, *arrs) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(arrs[0])
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        vals[i] = fn(*[a[idx] for a in arrs])
    return ci(vals)


def main() -> None:
    D.set_seed(SEED)
    random.seed(SEED)
    device = "cuda"
    log = (OUT / "finetune.log").open("w")
    items = load_gt()
    folds = make_folds(items)
    stems = [it[0] for it in items]
    gt = np.array([it[1] for it in items], float)
    det = np.array([it[2] for it in items], float)

    print(f"fine-tune density head w/ 2015-16 clicks | n={len(items)} "
          f"folds={FOLDS} epochs={EPOCHS} oversample={OVERSAMPLE} lr={LR} "
          f"sigma_out={SIGMA_OUT:.2f}", flush=True)
    log.write(f"n={len(items)} folds={FOLDS} epochs={EPOCHS} "
              f"oversample={OVERSAMPLE} lr={LR} sigma_out={SIGMA_OUT:.3f}\n")

    ft_pred = np.full(len(items), np.nan)
    clear_ret = None
    for k, test_idx in enumerate(folds):
        train_idx = [i for i in range(len(items)) if i not in set(test_idx)]
        train_stems = [stems[i] for i in train_idx]
        test_stems = [stems[i] for i in test_idx]
        print(f"  fold {k}: train2015={len(train_stems)} test2015={len(test_stems)}",
              flush=True)
        model = train_fold(train_stems, device, log)
        ft_pred[test_idx] = predict(model, test_stems, device)
        if k == 0:  # clear-val retention on the first fold's model
            mae, _ = D.evaluate(
                model,
                DataLoader(D.WormDensity("val", False), batch_size=8,
                           num_workers=WORKERS),
                device,
            )
            clear_ret = mae

    # base (pre-finetune) prediction on all 35 for a paired comparison
    base = D.CSRNet().to(device)
    base.load_state_dict(torch.load(D.OUT / "best.pt", map_location=device))
    base_pred = predict(base, stems, device)
    det_c = det / 0.72

    def mae(p, t):
        return np.abs(p - t).mean()

    def ratio(p, t):
        return p.sum() / t.sum()

    def pearson(p, t):
        return np.corrcoef(p, t)[0, 1]

    print(f"\nOOD 2015-16 results (n={len(gt)}, GT mean {gt.mean():.1f}/fr, "
          f"bootstrap 95% CI)\n")
    hdr = f"{'predictor':<26} {'r':>18} {'MAE':>18} {'ratio':>18}"
    print(hdr)
    log.write("\n" + hdr + "\n")
    for name, p in [
        ("density FT (held-out CV)", ft_pred),
        ("density base (pre-FT)", base_pred),
        ("detector/0.72", det_c),
    ]:
        rlo, rhi = boot(pearson, p, gt)
        mlo, mhi = boot(mae, p, gt)
        olo, ohi = boot(ratio, p, gt)
        line = (f"{name:<26} {pearson(p, gt):.3f} ({rlo:.2f},{rhi:.2f})  "
                f"{mae(p, gt):5.2f} ({mlo:.1f},{mhi:.1f})  "
                f"{ratio(p, gt):.2f}x ({olo:.2f},{ohi:.2f})")
        print(line)
        log.write(line + "\n")

    rline = f"\nclear-val retention (fold0 model): MAE {clear_ret:.2f} (base 2.12)"
    print(rline)
    log.write(rline + "\n")

    with (OUT / "ft2015_cv_results.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["stem", "gt", "detector", "base_density", "ft_density_heldout"])
        for i, (st, g, dc) in enumerate(items):
            wr.writerow([st, g, dc, f"{base_pred[i]:.2f}", f"{ft_pred[i]:.2f}"])
    print(f"\ncsv -> {OUT / 'ft2015_cv_results.csv'}")
    log.close()


if __name__ == "__main__":
    main()
