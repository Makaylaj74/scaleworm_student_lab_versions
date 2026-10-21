"""Build a 2021-2023 scale-worm series with the CLEAR density head (÷0.96).

Method-consistent companion to build_worm_series_2015.py / _2016.py so the whole
2015->2024 hybrid is counted by ONE method (CSRNet density head), not a density/
box-detector seam. Here the in-distribution clear model (csrnet_run/best.pt, OOD
held-out ratio-to-GT 0.96x) is run on the 2021-2023 clear recordings.

Frame selection is REUSED from the validated v8+SAHI coarse-max re-harvest
(harvest_clear_v8sahi_2021_2023.csv + harvest_janaug2021_v8sahi.csv): each
recording's best_time_s is the time that maximised the detector count over a
coarse 60 s grid -- the same frame-finding validated to ~100% of manual Scene-1
in aggregate. The COUNTER changes (density head, not detector); the FRAME does
not. So this series is method-consistent with 2015/2016 on the counter, and with
the 2021-2023 detector index on the frame choice.

Leakage: the clear model was trained on 2021-24 Monday manual frames, so a few
harvested Monday recordings may overlap its training set. Unlike the 2015/2016
case (in-sample 1.0x vs OOD 0.61x -- a 60% per-frame seam), the clear model reads
~0.96x on HELD-OUT frames too, so any overlap inflates by <~4% (within noise) and
needs no per-frame special-casing. Noted, not corrected.

Scale: counts ÷0.96 to a GT-referenced scale (matches the 2015/2016 ÷0.61
convention so the hybrid blocks sit on one scale). Conservative RELATIVE index,
not a calibrated absolute count. Resumable: re-running skips frame_ids already in
the per-frame CSV.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import density_head as D

REPO = Path("/home/jovyan/scaleworm-student-lab")
SAN = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
HARVESTS = [
    REPO / "notebooks/harvest_clear_v8sahi_2021_2023.csv",
    REPO / "notebooks/harvest_janaug2021_v8sahi.csv",
]
CLEAR_MODEL = REPO / "notebooks/csrnet_run/best.pt"
OUT = REPO / "notebooks/worm_series_2021_2023_density_run"
PERFRAME = OUT / "worm_series_2021_2023_density_per_frame.csv"
WEEKLY = OUT / "worm_series_2021_2023_density_weekly.csv"
RATIO = 0.96  # clear model OOD held-out ratio-to-GT; divide to GT-reference


def video_path(frame_id: str, dt: datetime) -> Path:
    return SAN / f"{dt:%Y/%m/%d}" / f"{frame_id}.mp4"


def load_harvest() -> list[dict]:
    seen: set[str] = set()
    recs: list[dict] = []
    for h in HARVESTS:
        for r in csv.DictReader(h.open()):
            fid = r["frame_id"]
            if fid in seen:
                continue
            seen.add(fid)
            dt = datetime.fromisoformat(r["datetime_utc"])
            recs.append({"frame_id": fid, "dt": dt,
                         "t": float(r["best_time_s"])})
    return sorted(recs, key=lambda x: x["dt"])


def extract_rgb(video: Path, t_s: float) -> Image.Image | None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


@torch.no_grad()
def density_count(model, img: Image.Image, device, norm) -> float:
    x = norm(transforms.functional.to_tensor(
        img.resize((D.IN_W, D.IN_H), Image.BILINEAR)))[None].to(device)
    return float(model(x).sum().item())


def load_done() -> dict[str, dict]:
    if not PERFRAME.exists():
        return {}
    return {r["frame_id"]: r for r in csv.DictReader(PERFRAME.open())}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    recs = load_harvest()
    done = load_done()
    print(f"2021-2023 clear recordings: {len(recs)}  "
          f"({len(done)} already done, {len(recs) - len(done)} to count)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = D.CSRNet().to(device)
    model.load_state_dict(torch.load(CLEAR_MODEL, map_location=device))
    model.eval()
    norm = transforms.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)

    rows: list[dict] = list(done.values())
    done_ids = set(done)
    fields = ["frame_id", "datetime_utc", "density_count", "worm_index"]
    n_new = n_fail = 0
    for i, rec in enumerate(recs):
        if rec["frame_id"] in done_ids:
            continue
        img = extract_rgb(video_path(rec["frame_id"], rec["dt"]), rec["t"])
        if img is None:
            n_fail += 1
            continue
        c = density_count(model, img, device, norm)
        rows.append({"frame_id": rec["frame_id"],
                     "datetime_utc": rec["dt"].isoformat(),
                     "density_count": round(c, 2),
                     "worm_index": round(c / RATIO, 2)})
        n_new += 1
        if n_new % 200 == 0:
            _write_perframe(rows, fields)
            print(f"  {n_new} counted ({i + 1}/{len(recs)} scanned, "
                  f"{n_fail} extract-fails)")
    _write_perframe(rows, fields)
    print(f"done: {n_new} new, {n_fail} extract-fails, {len(rows)} total")

    _write_weekly(rows)


def _write_perframe(rows: list[dict], fields: list[str]) -> None:
    rows.sort(key=lambda r: r["datetime_utc"])
    with PERFRAME.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def _write_weekly(rows: list[dict]) -> None:
    by_week: dict[str, list[float]] = {}
    for r in rows:
        d = datetime.fromisoformat(r["datetime_utc"]).replace(tzinfo=UTC).date()
        monday = d - timedelta(days=d.weekday())
        by_week.setdefault(monday.isoformat(), []).append(float(r["worm_index"]))
    with WEEKLY.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["week", "density_mean", "density_sem", "n_recordings"])
        for wk in sorted(by_week):
            v = np.array(by_week[wk])
            sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            wr.writerow([wk, f"{v.mean():.3f}", f"{sem:.4f}", len(v)])
    print(f"weekly -> {WEEKLY}  ({len(by_week)} weeks)")


if __name__ == "__main__":
    main()
