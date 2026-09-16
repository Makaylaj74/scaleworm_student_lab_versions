"""Period comparison of manual scaleworm abundance + a worms-vs-temperature figure.

Two products from the completed 2021-2024 manual Monday series
(``validation/monday_manual_series/monday_manual_timeseries.csv``) and the Mushroom
diffuse-flow thermistor array (``notebooks/ashes_thermistor_weekly_2017_2024.csv``):

1. A printed period comparison of worms/frame, reported under TWO splits:
   - calendar 2021-2022 vs 2023-2024 (as asked), and
   - clear-window vs blur-window at the 2023-08-10 camera-blur onset (the
     analytically meaningful split: countability, not the calendar, changes there).
   Each split reports the frame-weighted mean worms/frame with a 95% CI from a
   CLUSTER bootstrap over Mondays (the weekly sampling unit; frames within a Monday
   are pseudo-replicates), 10000 resamples, percentile method, seed 20260916.

2. ``notebooks/figure_worms_vs_temperature_manual.png`` -- 2-panel, shared x-axis:
   worms/frame (mean +/- SEM) over Mushroom diffuse-flow temperature.

INTEGRITY NOTES (baked into the caption too):
- Blur onset ~2023-08-10 degrades countability; post-onset counts are LOWER BOUNDS,
  so a lower 2024 mean is NOT by itself evidence of biological decline.
- ``array_max`` (hottest thermistor) is driven by a single sensor (#07 in 2021) --
  a localized diffuse-flow pulse, not a field-wide temperature; ``array_mean`` is the
  field-representative series. Both are shown; neither implies a causal worm link.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
NB = REPO / "notebooks"
WORM = REPO / "validation/monday_manual_series/monday_manual_timeseries.csv"
THERM = NB / "ashes_thermistor_weekly_2017_2024.csv"

BLUR_ONSET = pd.Timestamp("2023-08-10")  # documented camera-blur onset
N_BOOT = 10_000
SEED = 20260916

# Okabe-Ito
BLUE, ORANGE, VERMILLION, BLACK = "#0072B2", "#E69F00", "#D55E00", "#000000"


def cluster_bootstrap_mean(
    totals: np.ndarray, slots: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """95% CI for the frame-weighted mean worms/frame, resampling Mondays.

    Point estimate is ``sum(total_worms) / sum(n_slots)`` over the period's Mondays.
    Each bootstrap replicate resamples whole Mondays (rows) with replacement so the
    interval reflects week-to-week variation, not inflated per-frame n.
    """
    n = len(totals)
    reps = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        reps[i] = totals[idx].sum() / slots[idx].sum()
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def period_stats(df: pd.DataFrame, label: str, rng: np.random.Generator) -> dict:
    totals = df["total_worms"].to_numpy(float)
    slots = df["n_slots"].to_numpy(float)
    mean = totals.sum() / slots.sum()
    lo, hi = cluster_bootstrap_mean(totals, slots, rng)
    return {
        "period": label,
        "n_mondays": len(df),
        "n_frames": int(slots.sum()),
        "worms": int(totals.sum()),
        "mean_per_frame": round(mean, 2),
        "ci95_lo": round(lo, 2),
        "ci95_hi": round(hi, 2),
    }


def load_worms() -> pd.DataFrame:
    w = pd.read_csv(WORM, parse_dates=["date"])
    w["sem"] = pd.to_numeric(w["sem_worms"], errors="coerce")
    return w.sort_values("date").reset_index(drop=True)


XLIM = (pd.Timestamp("2021-01-01"), pd.Timestamp("2025-01-01"))


def plot_worm_panel(ax, w: pd.DataFrame, *, legend: bool = True) -> None:
    """Draw the manual worm-abundance panel (shared by all worm-vs-geophysics figs).

    Solid Okabe-Ito blue, mean ± SEM, connecting line broken across >14-day sampling
    gaps, with the camera-blur-onset marker and its 'counts→lower bound' note.
    """
    seg = w["date"].diff().dt.days.gt(14).cumsum()
    first = True
    for _, s in w.groupby(seg):
        ax.errorbar(
            s["date"],
            s["mean_worms"],
            yerr=s["sem"],
            color=BLUE,
            marker="o",
            ms=3,
            lw=1.0,
            elinewidth=0.6,
            capsize=1.5,
            label="Monday mean ± SEM (≤8 Scene-1 slots)" if first else None,
        )
        first = False
    ax.set_ylabel("scale-worms / frame\n(mean ± SEM)")
    ax.set_title(
        "Manual scale-worm abundance — Mushroom vent, Axial Seamount (2021–2024)",
        fontsize=11,
        loc="left",
    )
    ax.axvline(BLUR_ONSET, color=BLACK, ls="--", lw=1.0, alpha=0.7)
    ax.text(
        BLUR_ONSET,
        ax.get_ylim()[1] * 0.97,
        " camera-blur onset\n (counts→lower bound)",
        fontsize=7.5,
        color=BLACK,
        va="top",
        ha="left",
    )
    ax.grid(alpha=0.25, lw=0.5)
    if legend:
        ax.legend(fontsize=8, frameon=False, loc="upper right")


def compare(w: pd.DataFrame) -> list[dict]:
    rng = np.random.default_rng(SEED)
    splits = [
        ("2021-2022", w[w["date"] < "2023-01-01"]),
        ("2023-2024", w[w["date"] >= "2023-01-01"]),
        ("clear (<2023-08-10)", w[w["date"] < BLUR_ONSET]),
        ("blur (>=2023-08-10)", w[w["date"] >= BLUR_ONSET]),
    ]
    return [period_stats(sub, lbl, rng) for lbl, sub in splits]


def _plot(w: pd.DataFrame) -> None:
    therm = pd.read_csv(THERM, parse_dates=["week_start"])
    therm["week_start"] = therm["week_start"].dt.tz_localize(None)
    # crop to the display window so the y-axis scales to 2021-2024, not the
    # much larger 2017 thermistor spikes that would otherwise squash the panel.
    therm = therm[therm["week_start"].between(*XLIM)]

    fig, (ax_w, ax_t) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True)
    plot_worm_panel(ax_w, w)

    # --- temperature ---
    ax_t.plot(
        therm["week_start"],
        therm["array_max"],
        color=VERMILLION,
        lw=1.0,
        label="hottest thermistor (localized)",
    )
    ax_t.plot(
        therm["week_start"],
        therm["array_mean"],
        color=ORANGE,
        lw=1.3,
        label="array mean (field)",
    )
    ax_t.set_ylabel("diffuse-flow\ntemperature (°C)")
    ax_t.set_title(
        "Mushroom diffuse-flow temperature — OOI TMPSFA301 24-thermistor array",
        fontsize=11,
        loc="left",
    )
    ax_t.legend(fontsize=8, frameon=False, loc="upper right")
    # blur-onset marker + grid on the geophysics panel (worm panel already has them)
    ax_t.axvline(BLUR_ONSET, color=BLACK, ls="--", lw=1.0, alpha=0.7)
    ax_t.grid(alpha=0.25, lw=0.5)

    ax_t.set_xlim(*XLIM)
    ax_t.xaxis.set_major_locator(mdates.MonthLocator((1, 7)))
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_t.get_xticklabels(), rotation=45, ha="right")
    ax_t.set_xlabel("week (weekly Monday sampling)")

    cap = (
        "AI-generated caption (Claude, Anthropic) — for review. Top: manual "
        "box-corrected scale-worm (Polynoidae) counts at the Mushroom vent, Axial "
        "Seamount, from OOI HD video (CAMHDA301); each point = mean over that Monday's "
        "front-on Scene-1 slots (≤8), error bars ±1 SEM (absent when 1 slot). 128 "
        "Mondays, 2021-09-06..2024-12-23, 714 frames. Bottom: OOI TMPSFA301 24-thermistor "
        "diffuse-flow array, QARTOD-passed weekly mean; 'hottest thermistor' is a single "
        "sensor (localized pulse), not a field-wide temperature. Dashed line = ~2023-08-10 "
        "camera-blur onset; worm counts AFTER it are LOWER BOUNDS (reduced countability), "
        "so a lower 2024 mean is not by itself a biological decline. No worm–temperature "
        "correlation is asserted (worm sampling begins 2021-09, after the spring-2021 "
        "temperature spike). Derived from version-controlled scripts + data."
    )
    fig.text(
        0.01, 0.005, cap, fontsize=6.4, color="#444", wrap=True, ha="left", va="bottom"
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    out = NB / "figure_worms_vs_temperature_manual.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    w = load_worms()
    rows = compare(w)
    hdr = f"{'period':>22} {'Mon':>4} {'frames':>7} {'worms':>6} {'mean/fr':>8} {'95% CI':>16}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['ci95_lo']}, {r['ci95_hi']}]"
        print(
            f"{r['period']:>22} {r['n_mondays']:>4} {r['n_frames']:>7} {r['worms']:>6} "
            f"{r['mean_per_frame']:>8} {ci:>16}"
        )
    _plot(w)


if __name__ == "__main__":
    main()
