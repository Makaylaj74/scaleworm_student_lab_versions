"""Slide figure: hand-counted scale worms and their rate of change (dw/dt), Jan 2021 - Aug 2023.

PowerPoint companion to plot_worms_inflation_recount_2021_2023.py with the geophysics
panels removed. Same input series (hand counts under notebooks/worm_counting_rule.md:
Jan-Aug 2021 + Sep-Dec 2021 blind recounts, 2022-Aug 2023 original Monday series).

    (a) monthly mean worms per frame +/- 1 SEM, computed over that month's Monday means
        (the Monday is the sampling unit; frames within a Monday are not independent)
    (b) dw/dt = change in monthly mean from the previous month (worms per frame per
        month), only between consecutive months that both have data; error bars =
        sqrt(SEM_prev^2 + SEM_this^2), omitted when either month has a single Monday

No caption is drawn on the slide; it is written to the .caption.md file.

    .venv/bin/python3 scripts/plot_worm_dwdt_slide_2021_2023.py
"""

from __future__ import annotations

import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_worms_inflation_recount_2021_2023 import (
    END,
    JANAUG,
    JANAUG_REST,
    MAIN,
    NB,
    RULE_START,
    SEPDEC,
    START,
    weekly,
)

OUT = NB / "figure_worm_dwdt_slide_2021_2023"

# Okabe-Ito
BLUE, GREEN, VERMILLION = "#0072B2", "#009E73", "#D55E00"
# Presentation tier: title >=20, axis >=14, ticks >=12, lines >=1.5, dpi >=150
TITLE_PT, AXIS_PT, TICK_PT, PANEL_PT, LW = 22, 16, 14, 16, 2.2

CAPTION = (
    "AI-generated caption (Claude, Anthropic) — Hand-counted scale worms at Mushroom vent "
    "(ASHES, Axial Seamount) in clear OOI CAMHD footage (CAMHDA301 Scene-1), Jan 2021–Aug "
    "2023, all counted by M. Joseph under one written counting rule "
    "(notebooks/worm_counting_rule.md). Jan–Dec 2021 are Nov 2026 blind recounts; "
    "2022–Aug 2023 is the original Monday series (blind recount agreed, 0.95× [95% CI "
    "0.77–1.19]). (a) Monthly mean worms per frame ± 1 SEM over that month's Monday means. "
    "(b) dw/dt: change in the monthly mean from the previous month (green = increase, "
    "vermillion = decrease); error bars combine the two months' SEMs and are omitted when "
    "a month has a single Monday; months without data leave a gap and no dw/dt."
)


def monthly(weekly_means: pd.Series) -> pd.DataFrame:
    """Monthly mean and SEM over Monday means; one row per calendar month with data."""
    g = weekly_means.groupby(weekly_means.index.to_period("M")).agg(
        ["mean", "std", "size"]
    )
    g["sem"] = g["std"] / g["size"].map(math.sqrt)
    return g


def dwdt(m: pd.DataFrame) -> pd.DataFrame:
    """Month-to-month change in the monthly mean, consecutive months only."""
    prev = m.shift(1)
    consecutive = (m.index.to_series().diff() == 1 * m.index.freq).to_numpy()
    d = pd.DataFrame(
        {
            "dw": m["mean"] - prev["mean"],
            "err": np.sqrt(m["sem"] ** 2 + prev["sem"] ** 2),
        },
        index=m.index,
    )
    return d[consecutive]


def load() -> pd.DataFrame:
    w = pd.concat(
        [
            weekly([JANAUG, JANAUG_REST], START, RULE_START),
            weekly([SEPDEC], RULE_START, START + pd.DateOffset(years=1)),
            weekly([MAIN], pd.Timestamp("2022-01-01"), END),
        ]
    )["mean"]
    return monthly(w)


def draw() -> plt.Figure:
    m = load()
    d = dwdt(m)
    x = m.index.to_timestamp() + pd.Timedelta(days=14)
    full = pd.period_range(START, END, freq="M")
    mf = m.reindex(full)  # NaN rows break the line at empty months
    xf = full.to_timestamp() + pd.Timedelta(days=14)

    fig, (a, b) = plt.subplots(
        2, 1, figsize=(13.33, 7.5), sharex=True, gridspec_kw={"hspace": 0.12}
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.87, bottom=0.12)

    a.fill_between(
        xf, mf["mean"] - mf["sem"], mf["mean"] + mf["sem"], color=BLUE, alpha=0.25, lw=0
    )
    a.plot(xf, mf["mean"], color=BLUE, lw=LW, marker="o", ms=6)
    a.set_ylabel("Worms\n(per frame)", fontsize=AXIS_PT, fontweight="bold")
    a.set_ylim(0, math.ceil((m["mean"] + m["sem"].fillna(0)).max() / 5) * 5 + 2)

    xd = d.index.to_timestamp() + pd.Timedelta(days=14)
    colors = [GREEN if v >= 0 else VERMILLION for v in d["dw"]]
    b.bar(xd, d["dw"], width=22, color=colors, lw=0, zorder=3)
    b.errorbar(
        xd,
        d["dw"],
        yerr=d["err"],
        fmt="none",
        ecolor="#333333",
        lw=1.5,
        capsize=3,
        zorder=4,
    )
    b.axhline(0, color="black", lw=1.5, zorder=2)
    b.set_ylabel("dw/dt\n(per frame\nper month)", fontsize=AXIS_PT, fontweight="bold")
    lim = (d["dw"].abs() + d["err"].fillna(0)).max() * 1.1
    b.set_ylim(-lim, lim)

    pad = (END - START) * 0.03
    for i, ax in enumerate((a, b)):
        ax.set_xlim(START - pad, END + pad)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        ax.tick_params(labelsize=TICK_PT, width=1.5)
        ax.text(
            0.01,
            0.95,
            f"({'ab'[i]})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=PANEL_PT,
            fontweight="bold",
        )
    b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    b.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    b.set_xlabel("Date (monthly, UTC)", fontsize=AXIS_PT, fontweight="bold")
    fig.suptitle(
        "Mushroom vent scale worms and their rate of change, Jan 2021–Aug 2023",
        fontsize=TITLE_PT,
        fontweight="bold",
        y=0.97,
    )
    print(
        f"{len(m)} months, {len(d)} dw/dt points; x range {x.min():%Y-%m}..{x.max():%Y-%m}"
    )
    return fig


def main() -> None:
    fig = draw()
    fig.savefig(f"{OUT}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {OUT.name}.png")
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
