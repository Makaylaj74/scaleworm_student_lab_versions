"""Spot-check: are -2021's missed worms present-but-undetected, or a picking artifact?

Overlays Makayla's click positions (yellow x = worms she counted) and mushroom.pt
detections (red boxes, conf=0.25) on the three biggest-miss -2021 frames plus one
-2022 reference where the model does well. If clicks sit on clear worms that carry no
red box, the low -2021 recall is a genuine detector miss on sharp footage (domain/
framing), not an absence of worms or a bad scene pick.

Run under the thesis venv (ultralytics 8.4.62):
    /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python scripts/plot_2021_spotcheck.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from ultralytics import YOLO

REPO = Path("/home/jovyan/scaleworm-student-lab")
VAL = REPO / "validation/clear_window_handcount"
FRAMES = VAL / "frames"
CLICKS = VAL / "clicks"
RESULTS = VAL / "recall_gate_results.csv"
CONF = 0.25
RED = "#D55E00"
YEL = "#FFD400"


def sharpness(png):
    g = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def load_results():
    rows = []
    for r in csv.reader(RESULTS.open()):
        if not r or r[0].startswith("#") or r[0] == "frame_id":
            continue
        rows.append({"fid": r[0], "unit": r[1], "model": int(r[2]), "human": int(r[3])})
    return rows


def main() -> None:
    rows = load_results()
    r21 = sorted((r for r in rows if r["unit"] == "CAMHDA301-2021"),
                 key=lambda r: r["human"] - r["model"], reverse=True)
    ref = min((r for r in rows if r["unit"] == "CAMHDA301-2022" and r["human"] >= 8),
              key=lambda r: abs(r["human"] - r["model"]))
    picks = r21[:3] + [ref]

    model = YOLO(str(REPO / "mushroom.pt"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, rec in zip(axes.ravel(), picks):
        fid = rec["fid"]
        png = FRAMES / f"{fid}.png"
        img = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        # model boxes (red)
        res = model(str(png), conf=CONF, verbose=False)[0]
        nb = 0
        for x1, y1, x2, y2 in res.boxes.xyxy.tolist():
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                         fill=False, edgecolor=RED, linewidth=1.8))
            nb += 1
        # human clicks (yellow x)
        cp = CLICKS / f"{fid}.json"
        clicks = json.loads(cp.read_text())["clicks"] if cp.exists() else []
        if clicks:
            xs, ys = zip(*clicks)
            ax.scatter(xs, ys, s=70, marker="x", c=YEL, linewidths=2, zorder=5)
        unit = "−2021" if rec["unit"].endswith("2021") else "−2022"
        tag = "reference" if unit == "−2022" else "biggest misses"
        ax.set_title(f"{unit} ({tag})  ·  {fid}\nhuman {len(clicks)}  vs  model {nb}"
                     f"   ·  sharpness {sharpness(png):.0f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    handles = [mpatches.Patch(edgecolor=RED, facecolor="none", label="mushroom.pt detection"),
               plt.Line2D([], [], marker="x", color=YEL, linestyle="none", markersize=9,
                          markeredgewidth=2, label="worm counted by human")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=True)
    fig.suptitle("−2021 misses: worms are present (yellow ✕) but carry no detection (red) — "
                 "sharp footage, genuine detector miss", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = REPO / "notebooks/figure_2021_spotcheck_clicks_vs_boxes.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
    for rec in picks:
        print(f"  {rec['fid']}  {rec['unit']}  human={rec['human']} model={rec['model']}")


if __name__ == "__main__":
    main()
