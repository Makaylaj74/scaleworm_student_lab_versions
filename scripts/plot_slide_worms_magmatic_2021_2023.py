"""Half-slide vertical figure: two-year scale-worm record vs Axial magmatic activity.

Three stacked panels on a shared weekly (Monday) axis, Sep 2021 - Aug 2023 (the
validated clear-footage window of the automated record):

    (a) scale-worm abundance index (density head, weekly mean +/- SEM)
    (b) caldera uplift since the start of the window (OOI BOTPT, Central Caldera)
    (c) earthquakes per week above Mc (UW Axial cabled-array catalog)

Writes a slide version (no caption; the slide carries the talk text) and a captioned
version for the record, plus the caption text as Markdown for speaker notes.

    python scripts/plot_slide_worms_magmatic_2021_2023.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from plot_manual_timeseries import _justify

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks"
WORMS = NB / "hybrid_series_2015_2024.csv"
INFLATION = NB / "axial_inflation_weekly_2017_2024.csv"
SEISMICITY = NB / "axial_seismicity_weekly_2017_2024.csv"
OUT = NB / "figure_slide_worms_magmatic_2021_2023"

# Validated clear window of the automated series (pre-Sep-2021 has no manual truth).
START, END = pd.Timestamp("2021-09-01"), pd.Timestamp("2023-08-31")
REF_WEEKS = 2  # uplift re-referenced to the mean of the window's first 2 weeks

# Okabe-Ito
BLUE, VERMILLION, GREEN = "#0072B2", "#D55E00", "#009E73"
# Presentation tier: title >=20, axis >=14, ticks >=12, lines >=1.5, dpi >=150
TITLE_PT, AXIS_PT, TICK_PT, PANEL_PT, LW = 20, 14, 12, 14, 1.8

CAPTION = (
    "AI-generated caption (Claude, Anthropic) — Scale worms at Mushroom vent "
    "(ASHES, Axial Seamount) over two years of clear OOI CAMHD footage, Sep 2021–Aug "
    "2023, beside the magmatic re-inflation of Axial's caldera. (a) Worm abundance index: "
    "weekly mean (line) ± 1 SEM (band) of per-recording Scene-1 counts from the "
    "CSRNet density head (best.pt, scaled ÷0.96 to hand-count level; Mon–Fri, "
    "~3-hourly CAMHDA301 recordings; hybrid_series_2015_2024.csv). (b) Seafloor uplift at "
    "the Central Caldera BOTPT (OOI RS03CCAL-MJ03F-05-BOTPTA301, BOTSFLU daily de-tided "
    "depth, weekly mean), relative to the first two weeks of Sep 2021; gaps = missing "
    "weeks. (c) Earthquakes per week with M ≥ 0.25 (magnitude of completeness, "
    "max-curvature on this catalog) from the UW Axial cabled-array catalog (Wilcock et al. "
    "2016); counts are reduced during U.S. Navy high-rate data diversions. All series on "
    "Monday-start weeks. Uplift rose to ~12 cm by early 2023, then eased to ~8 cm; "
    "seismicity came in swarms, largest in winter 2021–22. The worm index ran ~10 "
    "per frame in late 2021, peaked at ~20–30 in late 2022, and fell back to ~13 by "
    "mid-2023. No correlation is tested here. Peak worm values are conservative "
    "(density ceiling at the densest mats)."
)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Weekly worm index, uplift and seismicity, trimmed to the window."""
    w = pd.read_csv(WORMS, parse_dates=["date"])
    w = w[(w.model == "best.pt") & w.date.between(START, END)].set_index("date")
    inf = pd.read_csv(INFLATION, parse_dates=["week_start"]).set_index("week_start")
    inf = inf.loc[START:END]
    ref = inf["uplift_cm"].dropna().iloc[:REF_WEEKS].mean()
    inf["uplift_window_cm"] = inf["uplift_cm"] - ref
    seis = pd.read_csv(SEISMICITY, parse_dates=["week_start"]).set_index("week_start")
    return w, inf, seis.loc[START:END]


def draw(with_caption: bool) -> plt.Figure:
    w, inf, seis = load()
    height = 11.1 if with_caption else 7.2
    fig = plt.figure(figsize=(6.4, height))
    top = 1 - 0.95 / height  # reserve ~0.95 in for the title
    plot_h = 5.25 / height  # leaves ~1 in below for 2-line date ticks + x label
    bottom = top - plot_h
    gap = 0.18 / height
    ph = (plot_h - 2 * gap) / 3
    left, width = 0.17, 0.78
    axes = [
        fig.add_axes([left, top - (i + 1) * ph - i * gap, width, ph]) for i in range(3)
    ]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    ax = axes[0]
    ax.fill_between(
        w.index,
        w.worm_index - w.spread,
        w.worm_index + w.spread,
        color=BLUE,
        alpha=0.25,
        lw=0,
        label="± 1 SEM",
    )
    ax.plot(w.index, w.worm_index, color=BLUE, lw=LW, label="Weekly mean")
    ax.set_ylabel("Worms\n(per frame)", fontsize=AXIS_PT, fontweight="bold")
    ax.set_ylim(0, 36)
    ax.legend(
        loc="upper left",
        fontsize=TICK_PT - 1,
        framealpha=0.9,
        ncol=2,
        handlelength=1.5,
        borderpad=0.3,
    )

    ax = axes[1]
    ax.plot(inf.index, inf.uplift_window_cm, color=VERMILLION, lw=LW)
    ax.set_ylabel("Uplift\n(cm)", fontsize=AXIS_PT, fontweight="bold")
    lo, hi = inf.uplift_window_cm.min(), inf.uplift_window_cm.max()
    ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))

    ax = axes[2]
    ax.bar(seis.index, seis.eq_count_mc, width=6, color=GREEN, lw=0)
    ax.set_ylabel("Earthquakes\n(per week)", fontsize=AXIS_PT, fontweight="bold")
    ax.set_ylim(0, seis.eq_count_mc.max() * 1.1)

    span = END - START
    pad = span * 0.03
    for i, ax in enumerate(axes):
        ax.set_xlim(START - pad, END + pad)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_linewidth(1.5)
        ax.tick_params(labelsize=TICK_PT, width=1.5)
        ax.text(
            0.985,
            0.94,
            f"({'abc'[i]})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=PANEL_PT,
            fontweight="bold",
        )
        if i < 2:
            ax.tick_params(labelbottom=False)
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(3, 9)))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[2].set_xlabel("Date (weekly, UTC)", fontsize=AXIS_PT, fontweight="bold")

    fig.text(
        0.5,
        1 - 0.18 / height,
        "Mushroom vent scale worms vs.\nAxial caldera inflation, 2021–2023",
        ha="center",
        va="top",
        fontsize=TITLE_PT,
        fontweight="bold",
    )
    if with_caption:
        _justify(fig, 0.06, bottom - 1.05 / height, 0.88, CAPTION, 10)
    return fig


def main() -> None:
    for cap, suffix in ((False, ""), (True, "_captioned")):
        fig = draw(cap)
        fig.savefig(f"{OUT}{suffix}.png", dpi=300)
        plt.close(fig)
        print(f"wrote {OUT.name}{suffix}.png")
    md = OUT.with_suffix(".caption.md")
    md.write_text(
        "*AI-generated draft (Claude, Anthropic) — for review. All parameters and "
        "figures are derived from version-controlled scripts and data.*\n\n"
        + CAPTION.split("— ", 1)[1]
        + "\n"
    )
    print(f"wrote {md.name}")


if __name__ == "__main__":
    main()
