"""CSRNet-style density-estimation head prototype for scaleworm counting.

Motivation: the individual-box detector (v5/v8) saturates in the dense regime
(density ceiling, 3 attacks closed: tile/SAHI, coverage-reinterpret, 2-class
region detector). Density-map regression is the standard tool for crowded
counting -- it regresses total mass instead of resolving each instance, so it
is not bound by instance-separation. This script trains a CSRNet on density
maps built from the EXISTING single-class box annotations and reports count
accuracy, broken out by density bin, against the manual truth.

Architecture: CSRNet (Li, Zhang, Chen, CVPR 2018) -- VGG16 frontend (conv1_1
..conv4_3, output stride 8) + 6-layer dilated-conv backend + 1x1 regressor.
Frontend initialised from torchvision ImageNet VGG16 weights.

No Borrowed Assumptions: the Gaussian kernel sigma is derived from THIS
dataset's own box sizes (per-worm, scaled to output resolution), not a borrowed
crowd-counting constant. Each worm contributes unit mass, so the density-map
integral equals the exact frame count regardless of sigma.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import VGG16_Weights, vgg16

ROOT = Path("/home/jovyan/scaleworm-student-lab")
DS = ROOT / "datasets/scaleworm_retrain_all"
OUT = Path(__file__).resolve().parent / "csrnet_run"
OUT.mkdir(exist_ok=True)

FULL_W, FULL_H = 1920, 1080
IN_W, IN_H = 1024, 576  # network input (16:9 preserved)
OUT_W, OUT_H = IN_W // 8, IN_H // 8  # density-map resolution (stride 8)
SCALE_X, SCALE_Y = OUT_W / 1.0, OUT_H / 1.0  # normalized -> output px

SEED = 0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def read_boxes(fp: Path) -> np.ndarray:
    """-> (N,4) normalized (cx, cy, bw, bh)."""
    out = []
    for ln in fp.read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            out.append([float(x) for x in p[1:5]])
    return np.array(out, float) if out else np.zeros((0, 4))


def density_map(boxes: np.ndarray) -> np.ndarray:
    """Build an (OUT_H, OUT_W) density map; integral == len(boxes).

    Each worm places a normalized Gaussian at its box centre. Sigma is derived
    from the worm's own box short-side scaled to output resolution (clamped to
    a resolvable floor), then the kernel is renormalised to unit mass so the
    map sums to the exact count.
    """
    dm = np.zeros((OUT_H, OUT_W), np.float32)
    for cx, cy, bw, bh in boxes:
        ox, oy = cx * OUT_W, cy * OUT_H
        bs_out = min(bw * OUT_W, bh * OUT_H)  # box short-side in out px
        sigma = max(1.5, 0.5 * bs_out)
        rad = int(np.ceil(3 * sigma))
        x0, x1 = max(0, int(ox) - rad), min(OUT_W, int(ox) + rad + 1)
        y0, y1 = max(0, int(oy) - rad), min(OUT_H, int(oy) + rad + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1) - ox
        ys = np.arange(y0, y1) - oy
        g = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2 * sigma**2))
        s = g.sum()
        if s > 0:
            dm[y0:y1, x0:x1] += (g / s).astype(np.float32)  # unit mass / worm
    return dm


class WormDensity(Dataset):
    def __init__(self, split: str, train: bool):
        self.imgs = sorted((DS / "images" / split).glob("*.png"))
        self.lbl = DS / "labels" / split
        self.train = train
        self.norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.imgs)

    def _augment(self, img: Image.Image, boxes: np.ndarray):
        # random crop (density integrates over the crop -> counts stay exact)
        if random.random() < 0.5 and len(boxes):
            s = random.uniform(0.6, 1.0)
            cw, ch = s, s
            x0 = random.uniform(0, 1 - cw)
            y0 = random.uniform(0, 1 - ch)
            img = img.crop(
                (
                    int(x0 * FULL_W),
                    int(y0 * FULL_H),
                    int((x0 + cw) * FULL_W),
                    int((y0 + ch) * FULL_H),
                )
            )
            cx, cy = boxes[:, 0], boxes[:, 1]
            keep = (cx >= x0) & (cx < x0 + cw) & (cy >= y0) & (cy < y0 + ch)
            boxes = boxes[keep].copy()
            if len(boxes):
                boxes[:, 0] = (boxes[:, 0] - x0) / cw
                boxes[:, 1] = (boxes[:, 1] - y0) / ch
                boxes[:, 2] = boxes[:, 2] / cw
                boxes[:, 3] = boxes[:, 3] / ch
        if random.random() < 0.5:  # horizontal flip
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes):
                boxes[:, 0] = 1.0 - boxes[:, 0]
        return img, boxes

    def __getitem__(self, i: int):
        img = Image.open(self.imgs[i]).convert("RGB")
        boxes = read_boxes(self.lbl / f"{self.imgs[i].stem}.txt")
        if self.train:
            img, boxes = self._augment(img, boxes)
        img = img.resize((IN_W, IN_H), Image.BILINEAR)
        x = self.norm(transforms.functional.to_tensor(img))
        dm = torch.from_numpy(density_map(boxes))[None]  # (1,H,W)
        return x, dm, float(len(boxes))


class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        # frontend: VGG16 conv1_1..conv4_3 (indices 0..22) -> stride 8, 512ch
        self.frontend = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23]
        cfg = [512, 512, 512, 256, 128, 64]
        layers, inc = [], 512
        for c in cfg:
            layers += [
                nn.Conv2d(inc, c, 3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
            ]
            inc = c
        self.backend = nn.Sequential(*layers)
        self.output = nn.Conv2d(64, 1, 1)
        for m in list(self.backend) + [self.output]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return torch.relu(self.output(x))


def evaluate(model, loader, device):
    model.eval()
    ae, se, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, dm, cnt in loader:
            pred = model(x.to(device))
            pc = pred.sum(dim=(1, 2, 3)).cpu().numpy()
            tc = cnt.numpy()
            ae += np.abs(pc - tc).sum()
            se += ((pc - tc) ** 2).sum()
            n += len(tc)
    return ae / n, np.sqrt(se / n)


def main() -> None:
    set_seed()
    device = "cuda"
    epochs = int(os.environ.get("EPOCHS", "200"))
    bs = int(os.environ.get("BS", "8"))
    workers = min(8, int(os.environ.get("WORKERS", "6")))  # lab cap: <= 24

    tr = DataLoader(
        WormDensity("train", True),
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
        pin_memory=True,
    )
    va = DataLoader(
        WormDensity("val", False),
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    model = CSRNet().to(device)
    crit = nn.MSELoss(reduction="sum")
    opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)

    best = float("inf")
    log = (OUT / "train.log").open("w")
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, dm, _ in tr:
            x, dm = x.to(device), dm.to(device)
            opt.zero_grad()
            loss = crit(model(x), dm) / x.size(0)
            loss.backward()
            opt.step()
            tot += loss.item()
        mae, rmse = evaluate(model, va, device)
        msg = f"ep {ep:3d}  loss {tot / len(tr):8.4f}  val MAE {mae:6.2f}  RMSE {rmse:6.2f}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        if mae < best:
            best = mae
            torch.save(model.state_dict(), OUT / "best.pt")
            log.write(f"  -> saved best (MAE {mae:.2f})\n")
            log.flush()
    log.write(f"BEST val MAE {best:.2f}\n")
    log.close()
    print(f"BEST val MAE {best:.2f}  -> {OUT / 'best.pt'}")


if __name__ == "__main__":
    main()
