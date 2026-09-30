"""Train a from-scratch Scene-1 tile classifier (CPU, no pretrained weights).

Binary classifier over the per-tile frames built by
``build_scene_classifier_dataset.py`` + ``extract_scene_classifier_frames.py``
(``datasets/scene_classifier/images/<split>/<label>/*.jpg``). The label dirs are
``1`` (the front-on Scene-1 view) and ``0`` (any other pan tile).

Why from-scratch: this host has no usable GPU (driver too old) and no network to
download pretrained backbone weights, so a small CNN trained on CPU is the honest
option. The net is deliberately compact (~0.2M params) with augmentation to fit
~1.7k positives without memorizing them.

Defensible-stats notes:
  - Splits are DAY-LEVEL (assigned upstream by build_scene_classifier_dataset.py,
    seed 20260922) so no same-day near-duplicate leaks across train/val/test.
  - Class imbalance (~3.3 neg per pos) handled with a pos_weight'd BCE loss.
  - The operating threshold is chosen on VALIDATION (max F1), never on test.
  - Test metrics are reported overall AND split by era (clear 2015/2021-22 vs
    blurry post-2023) because cross-era generalization is the real question and the
    eventual 2016 gate lives in the clear era.

Outputs (under --out, default models/scene_classifier/):
  best.pt          - state_dict of the best-val-F1 epoch + config
  metrics.json     - val + test metrics, chosen threshold, per-era test breakdown
  training_log.csv - per-epoch train/val loss + val F1/AUC
This script does NOT compute the recording-level gate; that is a separate step run
against the blind 2016 sort (all tiles of each recording, max-tile aggregation).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "datasets/scene_classifier/images"
OUT_DIR = REPO / "models/scene_classifier"
SEED = 20260930


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SmallCNN(nn.Module):
    """Compact 4-block conv net for 1-logit binary classification."""

    def __init__(self, widths: tuple[int, ...] = (16, 32, 64, 128), p_drop: float = 0.3):
        super().__init__()
        blocks: list[nn.Module] = []
        c_in = 3
        for c_out in widths:
            blocks += [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(widths[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


def make_loaders(
    data_root: Path, img_size: int, batch: int, workers: int, seed: int
) -> tuple[DataLoader, DataLoader, ImageFolder, ImageFolder]:
    norm = transforms.Normalize([0.5] * 3, [0.5] * 3)  # dataset-agnostic, no pretrain stats
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=6, translate=(0.06, 0.06), scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            norm,
        ]
    )
    eval_tf = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor(), norm]
    )
    train_ds = ImageFolder(str(data_root / "train"), transform=train_tf)
    val_ds = ImageFolder(str(data_root / "val"), transform=eval_tf)
    assert train_ds.class_to_idx == {"0": 0, "1": 1}, train_ds.class_to_idx
    g = torch.Generator()
    g.manual_seed(seed)
    train_dl = DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=workers, generator=g
    )
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers)
    return train_dl, val_dl, train_ds, val_ds


@torch.no_grad()
def predict(model: nn.Module, dl: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in dl:
        p = torch.sigmoid(model(x.to(device))).cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def prf(labels: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    pred = (probs >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), no sklearn dependency."""
    pos, neg = probs[labels == 1], probs[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # tie-averaged ranks (Mann-Whitney U)
    _, inv, counts = np.unique(probs, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    thrs = np.unique(np.concatenate([[0.0], np.sort(probs), [1.0]]))
    best_t, best_f = 0.5, -1.0
    for t in thrs:
        f = prf(labels, probs, t)["f1"]
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)  # <= 24-worker lab cap
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, train_ds, _ = make_loaders(
        args.data_root, args.img_size, args.batch, args.workers, args.seed
    )
    n_pos = sum(1 for _, y in train_ds.samples if y == 1)
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"device={device} train={len(train_ds)} (pos={n_pos} neg={n_neg}) "
          f"pos_weight={pos_weight.item():.2f}")

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    log_path = args.out / "training_log.csv"
    log_rows: list[dict] = []
    best_val_f1, best_state, bad = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(y)
            n += len(y)
        vp, vy = predict(model, val_dl, device)
        v_thr, _ = best_f1_threshold(vy, vp)
        vm = prf(vy, vp, v_thr)
        v_auc = roc_auc(vy, vp)
        row = {"epoch": epoch, "train_loss": tr_loss / n, "val_f1": vm["f1"],
               "val_precision": vm["precision"], "val_recall": vm["recall"],
               "val_auc": v_auc, "val_thr": v_thr}
        log_rows.append(row)
        print(f"[{epoch:02d}] loss={row['train_loss']:.4f} "
              f"val_f1={vm['f1']:.3f} P={vm['precision']:.3f} R={vm['recall']:.3f} "
              f"AUC={v_auc:.3f} thr={v_thr:.2f}")

        if vm["f1"] > best_val_f1:
            best_val_f1 = vm["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_thr = v_thr
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch} (no val-F1 gain in {args.patience})")
                break

    with log_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

    # ---- final test eval with the val-chosen model + threshold ----
    model.load_state_dict(best_state)
    eval_tf = val_dl.dataset.transform
    test_ds = ImageFolder(str(args.data_root / "test"), transform=eval_tf)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    tp_, ty_ = predict(model, test_dl, device)
    test_overall = prf(ty_, tp_, best_thr)
    test_overall["auc"] = roc_auc(ty_, tp_)

    # per-era test breakdown (era encoded in the source file path via labels.csv)
    era_by_path = _era_lookup(REPO / "datasets/scene_classifier/labels.csv")
    eras = np.array([era_by_path.get(Path(p).name, "unknown") for p, _ in test_ds.samples])
    per_era = {}
    for era in sorted(set(eras)):
        m = eras == era
        if m.sum():
            e = prf(ty_[m], tp_[m], best_thr)
            e["auc"] = roc_auc(ty_[m], tp_[m])
            e["n"] = int(m.sum())
            per_era[era] = e

    torch.save({"state_dict": best_state, "img_size": args.img_size,
                "threshold": best_thr, "seed": args.seed, "arch": "SmallCNN"},
               args.out / "best.pt")
    metrics = {"seed": args.seed, "img_size": args.img_size, "epochs_run": len(log_rows),
               "threshold": best_thr, "val_f1": best_val_f1,
               "test": test_overall, "test_by_era": per_era,
               "n_train": len(train_ds), "n_test": len(test_ds)}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("TEST:", json.dumps(test_overall))
    print("by era:", json.dumps(per_era))
    print(f"wrote {args.out/'best.pt'} + metrics.json")
    return 0


def _era_lookup(labels_csv: Path) -> dict[str, str]:
    """Map extracted-jpg filename -> era, so test tiles can be split by era."""
    out: dict[str, str] = {}
    with labels_csv.open(newline="") as fh:
        for r in csv.DictReader(fh):
            out[f"{r['stem']}_t{r['tile_time_s']}.jpg"] = r["era"]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
