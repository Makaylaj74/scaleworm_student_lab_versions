"""Build the 2015 early-recovery scale-worm series with the two-model density head.

This is the payoff: the first automated worm-abundance series for the 2015
post-Axial-eruption recovery (the project motivation). Counting uses the
2015-16 fine-tuned density model (OOD CV 0.61x); frame-finding is already
solved by MJ's manual scene sort (scene_sorting/full_2015: 120 Scene-1
recordings Jul-Dec 2015, each with scene1_time_s).

Leakage handling: 33 of the 120 Scene-1 frames were in the FT model's training
set. Running the deploy model on them would read in-sample (~1.0x) while fresh
frames read ~0.61x -- a seam inside the series. So those 33 use their held-out
5-fold CV prediction (ft2015_cv_results.csv); the other ~87 use the deploy
model (best_ft2015.pt). The whole series then reflects ~0.61x generalization
behavior consistently.

Scale: counts are divided by the model's overall ratio-to-GT (0.61) to put the
series on a GT-referenced scale. This is a conservative RELATIVE INDEX, not a
calibrated absolute count (dense OOD; single-rater GT ~2x uncertain per IRR;
the 0.61 scalar hides density-dependent spread -- sparse over, dense under).

Frame extraction matches the pipeline's canonical convention (POS_MSEC seek +
BGR->RGB), validated against an already-extracted anchor frame.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import density_head as D

REPO = Path("/home/jovyan/scaleworm-student-lab")
SORT = REPO / "scene_sorting/full_2015/sort_log.csv"
MANIFEST = REPO / "scene_sorting/full_2015/manifest.csv"
ANCHOR = REPO / "validation/subsample_2015_2016_stratified/frames"
CV_RESULTS = REPO / "notebooks/csrnet_ft2015_run/ft2015_cv_results.csv"
DEPLOY = REPO / "notebooks/csrnet_ft2015_run/best_ft2015.pt"
OUT = REPO / "notebooks/worm_series_2015_run"
FRAMES_OUT = OUT / "frames"
RATIO = 0.61  # FT model overall ratio-to-GT (OOD CV); divide to GT-reference


def load_scene1() -> list[dict]:
    man = {r["stem"]: r for r in csv.DictReader(MANIFEST.open())}
    out = []
    for r in csv.DictReader(SORT.open()):
        if r["decision"] == "scene1" and r.get("scene1_time_s"):
            m = man[r["stem"]]
            out.append({"stem": r["stem"], "t": float(r["scene1_time_s"]),
                        "date": m["recording_date"], "video": m["video_path"]})
    return sorted(out, key=lambda x: x["date"])


def extract_rgb(video: str, t_s: float) -> Image.Image | None:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def get_frame(rec: dict) -> Image.Image:
    """Reuse the already-extracted anchor frame if present, else extract."""
    anchor_fp = ANCHOR / f"{rec['stem']}.png"
    if anchor_fp.exists():
        return Image.open(anchor_fp).convert("RGB")
    dest = FRAMES_OUT / f"{rec['stem']}.png"
    if dest.exists():
        return Image.open(dest).convert("RGB")
    img = extract_rgb(rec["video"], rec["t"])
    if img is None:
        raise RuntimeError(f"extract failed: {rec['stem']}")
    img.save(dest)
    return img


def validate_extraction(recs: list[dict]) -> None:
    """Re-extract one anchor frame and confirm it matches the stored anchor."""
    for rec in recs:
        anchor_fp = ANCHOR / f"{rec['stem']}.png"
        if anchor_fp.exists():
            fresh = extract_rgb(rec["video"], rec["t"])
            stored = Image.open(anchor_fp).convert("RGB")
            diff = np.abs(np.asarray(fresh, int) - np.asarray(stored, int)).max()
            print(f"extraction check [{rec['stem']}]: max abs pixel diff = {diff}")
            if diff > 2:
                print("  WARNING: extraction does NOT match anchor (color/seek "
                      "mismatch) -- investigate before trusting counts")
            return
    print("extraction check: no overlapping anchor frame to validate against")


@torch.no_grad()
def density_count(model, img: Image.Image, device, norm) -> float:
    x = norm(transforms.functional.to_tensor(
        img.resize((D.IN_W, D.IN_H), Image.BILINEAR)))[None].to(device)
    return float(model(x).sum().item())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FRAMES_OUT.mkdir(exist_ok=True)
    recs = load_scene1()
    cv_pred = {r["stem"]: float(r["ft_density_heldout"])
               for r in csv.DictReader(CV_RESULTS.open())}
    print(f"2015 Scene-1 recordings: {len(recs)}  "
          f"({sum(r['stem'] in cv_pred for r in recs)} training->CV held-out, "
          f"{sum(r['stem'] not in cv_pred for r in recs)} fresh->deploy model)")

    validate_extraction(recs)

    device = "cuda"
    model = D.CSRNet().to(device)
    model.load_state_dict(torch.load(DEPLOY, map_location=device))
    model.eval()
    norm = transforms.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)

    rows = []
    for rec in recs:
        img = get_frame(rec)
        if rec["stem"] in cv_pred:
            count, src = cv_pred[rec["stem"]], "cv_heldout"
        else:
            count, src = density_count(model, img, device, norm), "deploy"
        rows.append({"stem": rec["stem"], "date": rec["date"],
                     "density_count": round(count, 2),
                     "worm_index": round(count / RATIO, 2), "source": src})

    rows.sort(key=lambda r: r["date"])
    per_frame = OUT / "worm_series_2015_per_frame.csv"
    with per_frame.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # daily aggregate (several Scene-1 recordings per data-day)
    by_day: dict[str, list[float]] = {}
    for r in rows:
        by_day.setdefault(r["date"], []).append(r["worm_index"])
    daily = OUT / "worm_series_2015_daily.csv"
    with daily.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["date", "n_frames", "worm_index_mean", "worm_index_std",
                     "worm_index_min", "worm_index_max"])
        for d in sorted(by_day):
            v = np.array(by_day[d])
            wr.writerow([d, len(v), f"{v.mean():.2f}",
                         f"{v.std(ddof=1) if len(v) > 1 else 0:.2f}",
                         f"{v.min():.2f}", f"{v.max():.2f}"])

    idx = np.array([r["worm_index"] for r in rows])
    print(f"\n2015 series built: {len(rows)} frames over {len(by_day)} data-days "
          f"({min(by_day)} -> {max(by_day)})")
    print(f"worm_index (GT-scale): mean {idx.mean():.1f}  "
          f"range {idx.min():.1f}-{idx.max():.1f}")
    print(f"per-frame -> {per_frame}\ndaily     -> {daily}")


if __name__ == "__main__":
    main()
