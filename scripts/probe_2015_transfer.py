"""Cheap OOD probe: does v5_all fire on 2015 Scene-1 (dense-mound early-recovery) frames?

Extracts the Scene-1 frame for a monthly spread of 2015 scene1 recordings, runs v5_all at
conf 0.25 and 0.40, and overlays boxes so we can eyeball whether the 2021-2024-trained
detector transfers to the 2015 scene at all. NOT a validation (no ground truth) -- a
qualitative transfer check before deciding light-calibration vs fine-tune-a-v6.
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SORT = REPO / "scene_sorting/full_2015/sort_log.csv"
MAN = REPO / "scene_sorting/full_2015/manifest.csv"
MODEL = REPO / "99_runs/scaleworm_v5_all/weights/best.pt"
OUT = REPO / "validation" / "probe_2015_v5_transfer.png"


def extract(video: str, t: float) -> Path | None:
    tmp = Path(tempfile.mkstemp(suffix=".png")[1])
    cmd = ["ffmpeg", "-y", "-ss", str(t), "-i", video, "-frames:v", "1", "-q:v", "2", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, check=False)
    return tmp if r.returncode == 0 and tmp.stat().st_size > 0 else None


def main() -> None:
    from ultralytics import YOLO

    scene1 = [r for r in csv.DictReader(SORT.open()) if r.get("decision") == "scene1"]
    vids = {r["stem"]: r["video_path"] for r in csv.DictReader(MAN.open())}
    # one recording per month, evenly across 2015
    by_month: dict[str, dict] = {}
    for r in scene1:
        key = r["stem"].split("-")[1][:6]  # YYYYMM
        by_month.setdefault(key, r)
    picks = [by_month[k] for k in sorted(by_month)][:6]

    model = YOLO(str(MODEL))
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()
    for ax, r in zip(axes, picks):
        stem = r["stem"]
        t = float(r.get("scene1_time_s") or 180)
        img = extract(vids[stem], t)
        if img is None:
            ax.set_title(f"{stem}: extract failed")
            ax.axis("off")
            continue
        res25 = model(str(img), conf=0.25, verbose=False)[0]
        res40 = model(str(img), conf=0.40, verbose=False)[0]
        ax.imshow(plt.imread(str(img)))
        for b in res25.boxes.xyxy.tolist():
            ax.add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                         fill=False, edgecolor="#D55E00", lw=1.0))
        for b in res40.boxes.xyxy.tolist():
            ax.add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                         fill=False, edgecolor="#009E73", lw=1.6))
        ax.set_title(f"{stem[11:19]}  boxes: {len(res40.boxes)}@0.40 / {len(res25.boxes)}@0.25",
                     fontsize=10)
        ax.axis("off")
        img.unlink(missing_ok=True)
    fig.legend(handles=[
        mpatches.Patch(color="#009E73", label="v5_all @ conf 0.40"),
        mpatches.Patch(color="#D55E00", label="v5_all @ conf 0.25 (adds these)"),
    ], loc="lower center", ncol=2, fontsize=11)
    fig.suptitle("v5_all on 2015 Scene-1 frames (OOD transfer probe — NO ground truth)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    print("picks:", [p["stem"] for p in picks])


if __name__ == "__main__":
    main()
