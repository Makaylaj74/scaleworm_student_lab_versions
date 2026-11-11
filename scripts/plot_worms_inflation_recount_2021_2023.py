"""Hand-counted scale worms vs Axial caldera inflation, Jan 2021 - Aug 2023 (post-recount).

Updates figure_slide_worms_magmatic_2021_2023 after the 2026-11-11 recounts: the worm
panel now shows MANUAL Monday counts (the advisor question was whether early 2021 is
really low), with the automated density-head index kept as context.

    (a) worms per frame, weekly (Monday) mean +/- SEM over that Monday's Scene-1 slots
        - Sep-Dec 2021: blind recount under the confirmed counting rule
          (validation/recount_sepdec_2021_rest; notebooks/worm_counting_rule.md)
        - 2022 - Aug 2023: original Monday series (blind 2022-24 recount = 0.95x)
        - Jan-Aug 2021: counted under the earlier BROAD criterion -> plotted hollow,
          provisional until recounted under the rule
        - automated density-head index (gray) for context
    (b) caldera uplift (OOI BOTPT) re-referenced to the first two weeks of Jan 2021
    (c) earthquakes per week above Mc (UW Axial cabled-array catalog)

    .venv/bin/python3 scripts/plot_worms_inflation_recount_2021_2023.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from plot_manual_timeseries import _justify

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks"
VAL = REPO / "validation"
MAIN = VAL / "monday_manual_series/frame_manifest.csv"
SEPDEC = VAL / "recount_sepdec_2021_rest/frame_manifest.csv"
JANAUG = VAL / "monday_janaug_2021/frame_manifest.csv"
AUTO = NB / "hybrid_series_2015_2024.csv"
INFLATION = NB / "axial_inflation_weekly_2017_2024.csv"
SEISMICITY = NB / "axial_seismicity_weekly_2017_2024.csv"
OUT = NB / "figure_worms_inflation_recount_2021_2023"

START, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2023-08-31")
RULE_START = pd.Timestamp("2021-09-01")  # first week counted under the confirmed rule
REF_WEEKS = 2

# Okabe-Ito
BLUE, ORANGE, VERMILLION, GREEN, GRAY = (
    "#0072B2",
    "#E69F00",
    "#D55E00",
    "#009E73",
    "#7F7F7F",
)
# Presentation tier: title >=20, axis >=14, ticks >=12, lines >=1.5, dpi >=150
TITLE_PT, AXIS_PT, TICK_PT, PANEL_PT, LW = 20, 14, 12, 14, 1.8

CAPTION = (
    "AI-generated caption (Claude, Anthropic) — Hand-counted scale worms at Mushroom vent "
    "(ASHES, Axial Seamount) in clear OOI CAMHD footage, Jan 2021–Aug 2023, beside the "
    "re-inflation of Axial's caldera. (a) Worms per frame: weekly (Monday) mean ± 1 SEM "
    "across that Monday's Scene-1 recordings (CAMHDA301, 3-hourly slots), box-corrected "
    "by M. Joseph. Blue = counted under the confirmed rule (notebooks/"
    "worm_counting_rule.md): Sep–Dec 2021 from the Nov 2026 blind recount (48 frames; "
    "1.24× the original Sep 2026 pass), 2022–Aug 2023 from the original Monday series "
    "(a blind 2022–24 recount agreed, 0.95× [95% CI 0.77–1.19]). Hollow orange = "
    "Jan–Aug 2021 (45 frames, 3 slots per Monday), counted under an earlier broader "
    "criterion that also boxed dark openings; provisional and likely high (the same "
    "criterion read 2.35× the rule on matched Sep–Dec 2021 Mondays) until recounted. "
    "Gray = automated CSRNet density-head index (÷0.96 to hand-count level; Mon–Fri; "
    "hybrid_series_2015_2024.csv), shown for context. (b) Seafloor uplift at the Central "
    "Caldera BOTPT (OOI RS03CCAL-MJ03F-05-BOTPTA301, de-tided daily depth, weekly mean) "
    "relative to the first two weeks of Jan 2021. (c) Earthquakes per week with M ≥ 0.25 "
    "(magnitude of completeness) from the UW Axial cabled-array catalog (Wilcock et al. "
    "2016). Gaps = weeks with no data. Under the rule, worms ran ~13 per frame in late "
    "2021, similar to early 2022, rose to a late-2022 peak, and fell by mid-2023: there is "
    "no low in late 2021. Whether Jan–Aug 2021 was low cannot be judged until it is "
    "recounted. No correlation with uplift or seismicity is tested here."
)


def weekly(manifest: Path, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """Per-Monday mean and SEM of counted Scene-1 frames in [lo, hi)."""
    m = pd.read_csv(manifest)
    m = m[m.frame_status == "counted"].copy()
    m["date"] = pd.to_datetime(m.frame_id.str[10:18])
    m = m[(m.date >= lo) & (m.date < hi)]
    m["n"] = m.worm_count.astype(float)
    g = m.groupby("date").n.agg(["mean", "std", "size"])
    g["sem"] = g["std"] / g["size"].map(math.sqrt)
    return g


def on_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to every Monday in the window so missing weeks plot as gaps."""
    weeks = pd.date_range(START, END, freq="W-MON")
    return df.reindex(weeks)


def load() -> dict[str, pd.DataFrame]:
    rule = pd.concat(
        [
            weekly(SEPDEC, RULE_START, START + pd.DateOffset(years=1)),
            weekly(MAIN, pd.Timestamp("2022-01-01"), END),
        ]
    )
    broad = weekly(JANAUG, START, RULE_START)
    auto = pd.read_csv(AUTO, parse_dates=["date"])
    auto = auto[(auto.model == "best.pt") & auto.date.between(START, END)]
    inf = pd.read_csv(INFLATION, parse_dates=["week_start"]).set_index("week_start")
    inf = inf.loc[START:END].copy()
    inf["uplift_window_cm"] = (
        inf["uplift_cm"] - inf["uplift_cm"].dropna().iloc[:REF_WEEKS].mean()
    )
    seis = pd.read_csv(SEISMICITY, parse_dates=["week_start"]).set_index("week_start")
    return {
        "rule": on_weeks(rule),
        "broad": on_weeks(broad),
        "auto": auto.set_index("date"),
        "inf": inf,
        "seis": seis.loc[START:END],
    }


def draw(with_caption: bool) -> plt.Figure:
    d = load()
    W = 10.0
    height = 11.7 if with_caption else 8.4
    fig = plt.figure(figsize=(W, height))
    top = 1 - 0.95 / height
    plot_h = 6.4 / height
    bottom = top - plot_h
    gap = 0.2 / height
    ph = (plot_h - 2 * gap) / 3
    left, width = 0.12, 0.84
    axes = [
        fig.add_axes([left, top - (i + 1) * ph - i * gap, width, ph]) for i in range(3)
    ]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    ax = axes[0]
    a = d["auto"]
    ax.plot(
        a.index,
        a.worm_index,
        color=GRAY,
        lw=LW,
        alpha=0.8,
        label="Automated index (context)",
    )
    r = d["rule"]
    ax.fill_between(
        r.index,
        r["mean"] - r["sem"],
        r["mean"] + r["sem"],
        color=BLUE,
        alpha=0.25,
        lw=0,
    )
    ax.plot(
        r.index,
        r["mean"],
        color=BLUE,
        lw=LW,
        marker="o",
        ms=4,
        label="Hand count, confirmed rule (± 1 SEM)",
    )
    b = d["broad"]
    ax.errorbar(
        b.index,
        b["mean"],
        yerr=b["sem"],
        fmt="o",
        ms=6,
        mfc="white",
        mec=ORANGE,
        mew=1.8,
        ecolor=ORANGE,
        elinewidth=1.5,
        capsize=0,
        label="Jan–Aug 2021, broad criterion (provisional)",
    )
    ax.axvline(RULE_START, color="k", ls=":", lw=1.5)
    ax.set_ylabel("Worms\n(per frame)", fontsize=AXIS_PT, fontweight="bold")
    ax.set_ylim(0, 62)
    ax.legend(
        loc="upper right",
        fontsize=TICK_PT - 1,
        framealpha=0.9,
        ncol=1,
        handlelength=1.8,
        borderpad=0.3,
    )

    ax = axes[1]
    ax.plot(d["inf"].index, d["inf"].uplift_window_cm, color=VERMILLION, lw=LW)
    ax.set_ylabel("Uplift\n(cm)", fontsize=AXIS_PT, fontweight="bold")
    lo, hi = d["inf"].uplift_window_cm.min(), d["inf"].uplift_window_cm.max()
    ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))

    ax = axes[2]
    s = d["seis"]
    ax.bar(s.index, s.eq_count_mc, width=6, color=GREEN, lw=0)
    ax.set_ylabel("Earthquakes\n(per week)", fontsize=AXIS_PT, fontweight="bold")
    ax.set_ylim(0, s.eq_count_mc.max() * 1.1)

    pad = (END - START) * 0.03
    for i, ax in enumerate(axes):
        ax.set_xlim(START - pad, END + pad)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_linewidth(1.5)
        ax.tick_params(labelsize=TICK_PT, width=1.5)
        ax.text(
            0.012,
            0.94,
            f"({'abc'[i]})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=PANEL_PT,
            fontweight="bold",
        )
        if i < 2:
            ax.tick_params(labelbottom=False)
    axes[0].text(
        RULE_START + pd.Timedelta(days=6),
        59,
        "confirmed rule →",
        fontsize=TICK_PT - 1,
        fontweight="bold",
        va="top",
    )
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[2].set_xlabel("Date (weekly, UTC)", fontsize=AXIS_PT, fontweight="bold")

    fig.text(
        0.5,
        1 - 0.18 / height,
        "Mushroom vent scale worms (hand counts) vs.\n"
        "Axial caldera inflation, Jan 2021–Aug 2023",
        ha="center",
        va="top",
        fontsize=TITLE_PT,
        fontweight="bold",
    )
    if with_caption:
        _justify(fig, 0.06, bottom - 1.0 / height, 0.88, CAPTION, 10)
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
