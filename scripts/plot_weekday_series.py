"""Combined Monday + Tuesday manual scale-worm series (2021-2024) on one figure.

Both weekday series are hand-corrected Scene-1 counts by MJ. To keep them
comparable on a single axis, per-day uncertainty for BOTH is recomputed with the
same percentile-bootstrap 95% CI used for the Tuesday series (build_tuesday_series
.aggregate_manual), read straight from each series' frame manifest -- so the Monday
points here carry a bootstrap CI, not the SEM stored in its older timeseries CSV.

The scientific point (confirmed, per results_note_detector_recall_2026_09_13): manual
abundance persists across the ~Aug-2023 camera-blur onset on both weekdays -- the
post-2023 collapse in the automated mushroom.pt index was a detection artifact.

Output: notebooks/figure_weekday_manual_series_2021_2024.png
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from build_tuesday_series import aggregate_manual
from plot_tuesday_series import BLUE, GREY, VERMILLION, _break_gaps, _justify

REPO = Path(__file__).resolve().parent.parent
MON_MANIFEST = REPO / "validation/monday_manual_series/frame_manifest.csv"
TUE_MANIFEST = REPO / "validation/tuesday_manual_series/box_correct_manifest.csv"
OUT_PNG = REPO / "notebooks/figure_weekday_manual_series_2021_2024.png"

BLUR_ONSET = date(2023, 8, 10)


def _series(df):
    dts = [datetime.fromisoformat(r) for r in df["date"]]
    mean = df["mean_worms"].to_numpy(dtype=float)
    lo = np.array([float(v) if v not in ("", "nan") else np.nan for v in df["ci_lo"]])
    hi = np.array([float(v) if v not in ("", "nan") else np.nan for v in df["ci_hi"]])
    return dts, mean, lo, hi


def main() -> None:
    mon = aggregate_manual(MON_MANIFEST)
    tue = aggregate_manual(TUE_MANIFEST)
    md, mm, mlo, mhi = _series(mon)
    td, tm, tlo, thi = _series(tue)

    fig, ax = plt.subplots(figsize=(13, 5.6))

    # connecting lines break across gaps > 14 days (no interpolation over gaps)
    for dts, mean, color in ((md, mm, BLUE), (td, tm, VERMILLION)):
        sx, sy = _break_gaps(dts, mean, max_gap_days=14)
        ax.plot(sx, sy, color=color, lw=1.1, alpha=0.5, zorder=2)

    ax.errorbar(
        md,
        mm,
        yerr=np.vstack([mm - mlo, mhi - mm]),
        fmt="o",
        ms=4.5,
        color=BLUE,
        ecolor=GREY,
        elinewidth=0.9,
        capsize=2.0,
        zorder=3,
        label=f"Monday hand count (n={len(md)} Mondays)",
    )
    ax.errorbar(
        td,
        tm,
        yerr=np.vstack([tm - tlo, thi - tm]),
        fmt="s",
        ms=4.5,
        color=VERMILLION,
        ecolor=VERMILLION,
        elinewidth=0.9,
        capsize=2.0,
        alpha=0.9,
        zorder=4,
        label=f"Tuesday hand count (n={len(td)} Tuesdays)",
    )

    # blur-onset context marker (vertical label in the post-peak gap, clear of the legend)
    ax.axvline(BLUR_ONSET, color=GREY, ls="--", lw=1.3, zorder=1)
    ax.text(
        BLUR_ONSET + timedelta(days=10),
        ax.get_ylim()[1] * 0.55,
        "camera blur onset (~Aug 2023)",
        color=GREY,
        fontsize=9,
        fontweight="bold",
        va="center",
        ha="left",
        rotation=90,
    )

    ax.set_ylabel("Scale-worm count per Scene-1 frame", fontweight="bold", fontsize=12)
    ax.set_xlabel("Date", fontweight="bold", fontsize=12)
    ax.set_title(
        "Manual scale-worm abundance — Mushroom vent, Axial Seamount, 2021-2024\n"
        "(hand-corrected Scene-1 counts; weekly Monday & Tuesday sampling, mean +/- 95% CI)",
        fontsize=13,
    )

    alldts = md + td
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    span = (max(alldts) - min(alldts)).days
    pad = timedelta(days=int(span * 0.02))
    ax.set_xlim(min(alldts) - pad, max(alldts) + pad)
    ax.grid(alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)
    for s in ax.spines.values():
        s.set_linewidth(1.0)
    ax.legend(framealpha=0.92, loc="upper left", fontsize=10)

    cap = (
        "AI-generated caption (Claude, Anthropic) — Manual scale-worm (Polynoidae) abundance at the "
        "Mushroom vent, Axial Seamount, from OOI Cabled Array HD video (instrument CAMHDA301, "
        "RS03ASHS-PN03B-06), 2021-2024. Blue circles are the weekly-Monday series (128 Mondays, 714 "
        "Scene-1 frames, 2021-09 to 2024-12); vermillion squares are the weekly-Tuesday series (49 "
        "Tuesdays, 229 frames, 2021-12 to 2023-08). Each point is the mean worm count across that "
        "day's front-on Scene-1 slots (up to eight three-hourly UTC slots); error bars are a "
        "percentile bootstrap 95% CI over the counted slots (n=10000, seed 20260922; absent for "
        "single-slot days). Counts are human box-corrected annotations (annotator MJ; worm_count = "
        "number of boxes); unusable-blur and non-front-on slots are excluded as missing data, not "
        "zeros, so each connecting line breaks across gaps > 14 days. The dashed line marks the "
        "~Aug-2023 onset of persistent camera blur; abundance does not collapse across it on either "
        "weekday, confirming the post-Sep-2023 drop in the automated mushroom.pt index was a "
        "detection artifact. Derived from validation/{monday,tuesday}_manual_series frame manifests "
        "via scripts/build_tuesday_series.py (aggregate_manual) and scripts/plot_weekday_series.py."
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.44)
    _justify(fig, 0.07, 0.25, 0.91, cap, fontsize=8)

    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(
        f"wrote {OUT_PNG.name}  Mondays={len(md)} Tuesdays={len(td)}  "
        f"Mon range {mm.min():.1f}-{mm.max():.1f}  Tue range {tm.min():.1f}-{tm.max():.1f}"
    )


if __name__ == "__main__":
    main()
