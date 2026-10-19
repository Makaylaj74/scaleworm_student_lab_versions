"""2015: box-detector (v8+SAHI) vs density-head counts, with hand-count GT reference.

Shows why the density head was necessary. The box detector (v8 tiled + SAHI,
the best box config) is run on the same 120 Scene-1 frames as the 2015 density
series. Hand-count GT (42 anchor frames) is overlaid as the truth reference.

Honest scaling: the density-head series uses its CV-validated ratio-to-GT
correction (/0.61). The detector's only correction is the 2021-24 clear-window
/0.72, which does NOT transfer to 2015 (its 2015 OOD ratio-to-GT is ~0.23), so
the detector is shown as its RAW count -- even /0.72 would read ~4x low. GT
points make the comparison unambiguous regardless of correction choice.

v8+SAHI config matches the production pipeline (conf 0.40, slice 640, overlap
0.2).
"""

from __future__ import annotations

import csv
import os
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/jovyan/scaleworm-student-lab")
V8 = REPO / "99_runs/scaleworm_v8_tiled/weights/best.pt"
PERFRAME = REPO / "notebooks/worm_series_2015_run/worm_series_2015_per_frame.csv"
ANCHOR = REPO / "validation/subsample_2015_2016_stratified/frames"
HANDCOUNT = REPO / "validation/subsample_2015_2016_stratified/handcount_sheet.csv"
RUN_FRAMES = REPO / "notebooks/worm_series_2015_run/frames"
OUT = REPO / "notebooks/worm_series_2015_run"
DENS_RATIO = 0.61  # density-head CV ratio-to-GT (already applied in worm_index)

CONF = float(os.environ.get("CONF", "0.40"))
SLICE = int(os.environ.get("SLICE", "640"))
OVERLAP = float(os.environ.get("OVERLAP", "0.2"))
C_DENS, C_DET, C_GT = "#0072B2", "#D55E00", "#000000"  # Okabe-Ito blue/orange

DISCLOSURE = ("AI-generated draft (Claude, Anthropic) — for review. All values "
              "derive from version-controlled scripts and data.")


def frame_path(stem: str) -> Path:
    for d in (ANCHOR, RUN_FRAMES):
        p = d / f"{stem}.png"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def run_detector() -> dict[str, int]:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    det = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(V8),
        confidence_threshold=CONF, device="cuda:0")
    stems = [r["stem"] for r in csv.DictReader(PERFRAME.open())]
    out = {}
    for i, stem in enumerate(stems):
        r = get_sliced_prediction(
            str(frame_path(stem)), det, slice_height=SLICE, slice_width=SLICE,
            overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP, verbose=0)
        out[stem] = len(r.object_prediction_list)
        if i % 20 == 0:
            print(f"  detector {i}/{len(stems)}", flush=True)
    return out


def daily(dates: list[str], vals: list[float]) -> tuple[list, np.ndarray, np.ndarray]:
    by: dict[str, list[float]] = {}
    for d, v in zip(dates, vals):
        by.setdefault(d, []).append(v)
    days = sorted(by)
    mean = np.array([np.mean(by[d]) for d in days])
    std = np.array([np.std(by[d], ddof=1) if len(by[d]) > 1 else 0 for d in days])
    dt = [datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=UTC) for d in days]
    return dt, mean, std


def main() -> None:
    rows = list(csv.DictReader(PERFRAME.open()))
    cache = OUT / "detector_vs_density_2015_per_frame.csv"
    if cache.exists():  # reuse cached detector counts (expensive SAHI inference)
        det = {r["stem"]: int(r["detector_count"])
               for r in csv.DictReader(cache.open())}
        print(f"loaded cached detector counts (n={len(det)})")
    else:
        det = run_detector()
        with cache.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["stem", "date", "detector_count", "density_index_gtscale"])
            for r in rows:
                wr.writerow([r["stem"], r["date"], det[r["stem"]], r["worm_index"]])

    dates = [r["date"] for r in rows]
    dens_idx = [float(r["worm_index"]) for r in rows]        # GT-scale (÷0.61)
    det_raw = [float(det[r["stem"]]) for r in rows]          # raw box count

    # GT hand-counts (2015 frames only), as truth reference points
    gt_d, gt_v = [], []
    for r in csv.DictReader(HANDCOUNT.open()):
        wc = (r["worm_count"] or "").strip()
        if wc and r["frame_id"] in det and r["year"] == "2015":
            gt_d.append(datetime.strptime(r["frame_id"][10:18], "%Y%m%d")
                        .replace(tzinfo=UTC))
            gt_v.append(float(wc))

    dd, dmean, dstd = daily(dates, dens_idx)
    xd, xmean, xstd = daily(dates, det_raw)

    fig, ax = plt.subplots(figsize=(11, 6.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.30)

    ax.errorbar(dd, dmean, yerr=dstd, fmt="o-", color=C_DENS, lw=1.6, ms=6,
                capsize=3, elinewidth=1.0, zorder=4,
                label="density head ÷0.61 (GT-scale), daily mean ± 1 SD")
    ax.errorbar(xd, xmean, yerr=xstd, fmt="s-", color=C_DET, lw=1.6, ms=5,
                capsize=3, elinewidth=1.0, zorder=3,
                label="v8+SAHI detector, raw count, daily mean ± 1 SD")
    ax.scatter(gt_d, gt_v, s=70, marker="*", color=C_GT, zorder=5,
               label=f"hand-count GT (n={len(gt_v)} frames)")

    ax.set_xlabel("Date (2015)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Scale-worm count (worms / Scene-1 frame)",
                  fontsize=13, fontweight="bold")
    ax.set_title("2015: box detector vs density head vs hand-count truth\n"
                 "Axial Seamount — ASHES vent field (Mushroom), CAMHDA301",
                 fontsize=14, fontweight="bold")

    allx = dd + xd + gt_d
    span = (max(allx) - min(allx)).days
    pad = span * 0.03
    ax.set_xlim(mdates.date2num(min(allx)) - pad, mdates.date2num(max(allx)) + pad)
    ax.set_ylim(0, max(max(gt_v), (dmean + dstd).max()) * 1.10)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3, zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    ax.legend(fontsize=9.5, framealpha=0.9, loc="upper right")

    det_mean = np.mean(det_raw)
    caption = (
        f"{DISCLOSURE}\n"
        f"2015 Scene-1 frames (120 frames, 21 data-days): box detector (v8 tiled "
        f"+ SAHI, conf {CONF}/slice {SLICE}/overlap {OVERLAP}; raw mean "
        f"{det_mean:.1f}/fr) reads ~4-5× below hand-count GT (mean 25.2/fr); the "
        "density head (÷0.61 CV-validated) tracks GT. The detector's 2021-24 "
        "clear-window ÷0.72 correction does NOT transfer to 2015 (OOD ratio-to-GT "
        "≈0.23) — even corrected it reads ~4× low, so raw count is shown. GT stars "
        "= MJ hand-counts (anchor frames). This is why the density head, not the "
        "box detector, counts the 2015 recovery. Source: "
        "build_detector_vs_density_2015.py."
    )
    fig.text(0.015, 0.005, caption, fontsize=7.5, family="monospace",
             va="bottom", ha="left", wrap=True)

    out_png = OUT / "figure_detector_vs_density_2015.png"
    fig.savefig(out_png, dpi=300)
    print(f"\ndetector raw mean {det_mean:.1f}/fr  vs  density GT-scale mean "
          f"{np.mean(dens_idx):.1f}/fr  vs  GT mean 25.2/fr")
    print(f"figure -> {out_png}")


if __name__ == "__main__":
    main()
