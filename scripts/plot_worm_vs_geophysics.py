"""Three separate 2-panel figures: scaleworm abundance (top) vs each geophysical
parameter (bottom), 2021-2024, on the shared weekly-Monday grid.

  fig 1: worms + Mushroom diffuse-flow temperature
  fig 2: worms + Axial seismicity
  fig 3: worms + caldera magma inflation

Worm panel distinguishes 2023-2024 MANUAL box-corrected counts (solid, filled
markers, error bars) from 2021-2022 PROVISIONAL v2 detector counts (open markers,
dashed) which under-count (~66-90% clear recall) and are a lower bound, not
level-comparable across the method boundary until box-corrected.
matplotlib 300 DPI, Okabe-Ito, gap-aware.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

NB = Path(__file__).resolve().parent.parent / "notebooks"
WORM = NB / "worm_timeseries_provisional_2017_2024.csv"
MC_TXT = NB / "axial_seismicity_weekly_2017_2024.mc.txt"

PURPLE, ORANGE, VERMILLION, BLUE, GREEN, BLACK = (
    "#CC79A7",
    "#E69F00",
    "#D55E00",
    "#0072B2",
    "#009E73",
    "#000000",
)


def _geo(name: str) -> pd.DataFrame:
    df = pd.read_csv(NB / name, index_col="week_start", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    grid = pd.date_range("2017-01-02", "2024-12-30", freq="W-MON")
    return df.reindex(grid)


def _plot_worms(ax) -> None:
    w = pd.read_csv(WORM, parse_dates=["date"])
    w["sem"] = pd.to_numeric(w["sem_worms"], errors="coerce")
    for method, style in [
        (
            "manual",
            {
                "color": PURPLE,
                "marker": "o",
                "ms": 3,
                "ls": "-",
                "label": "manual box-count (2023–2024)",
            },
        ),
        (
            "v2_provisional",
            {
                "color": PURPLE,
                "marker": "o",
                "mfc": "white",
                "ms": 3,
                "ls": "--",
                "alpha": 0.75,
                "label": "v2 provisional (2021–2022, undercounts)",
            },
        ),
    ]:
        sub = w[w["method"] == method].sort_values("date")
        # break line at >14-day gaps
        gap = sub["date"].diff().dt.days.gt(14)
        seg = gap.cumsum()
        for _, s in sub.groupby(seg):
            ax.errorbar(
                s["date"],
                s["mean_worms"],
                yerr=s["sem"],
                elinewidth=0.6,
                capsize=1.5,
                **style,
            )
            style = {k: v for k, v in style.items() if k != "label"}
    ax.set_ylabel("worms / frame\n(mean ± SEM)")
    ax.set_title(
        "Scaleworm abundance (Mushroom Scene-1, manual box-counts)",
        fontsize=10,
        loc="left",
    )
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper left")


def _finish(fig, ax_bottom, fname: str, extra_caption: str) -> None:
    ax_bottom.set_xlim(pd.Timestamp("2017-01-01"), pd.Timestamp("2025-01-01"))
    ax_bottom.xaxis.set_major_locator(mdates.YearLocator())
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_bottom.set_xlabel("week (2017–2024)")
    cap = (
        "AI-generated caption (Claude, Anthropic) — Weekly-Monday grid. "
        "Worms: Mushroom Scene-1 counts; 2023–2024 = manual box-corrected "
        "(ground truth), 2021–2022 = PROVISIONAL v2 detector (open markers, "
        "~66–90% clear recall → lower bound, not level-comparable until "
        "box-corrected). " + extra_caption
    )
    fig.text(
        0.01, 0.005, cap, fontsize=6.6, color="#444", wrap=True, ha="left", va="bottom"
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    out = NB / fname
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    mc = MC_TXT.read_text().strip() if MC_TXT.exists() else "?"

    # fig 1: temperature
    therm = _geo("ashes_thermistor_weekly_2017_2024.csv")
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    _plot_worms(ax[0])
    ax[1].plot(
        therm.index,
        therm["array_max"],
        color=VERMILLION,
        lw=1.2,
        label="hottest thermistor",
    )
    ax[1].plot(
        therm.index, therm["array_mean"], color=ORANGE, lw=1.1, label="array mean"
    )
    ax[1].set_ylabel("temperature (°C)")
    ax[1].set_title(
        "Mushroom diffuse-flow temperature (TMPSFA301)", fontsize=10, loc="left"
    )
    ax[1].legend(fontsize=7.5, ncol=2, frameon=False)
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle(
        "Scaleworm abundance vs diffuse-flow temperature, 2017–2024", fontsize=12
    )
    _finish(
        fig,
        ax[1],
        "figure_worms_vs_temperature.png",
        "Temperature: OOI TMPSFA301 24-thermistor array, QARTOD-passed.",
    )

    # fig 2: seismicity
    seis = _geo("axial_seismicity_weekly_2017_2024.csv")
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    _plot_worms(ax[0])
    ax[1].plot(
        seis.index, seis["eq_count_all"], color=BLUE, lw=1.1, label="all located events"
    )
    ax[1].plot(
        seis.index, seis["eq_count_mc"], color=BLACK, lw=0.9, ls=":", label=f"MW ≥ {mc}"
    )
    ax[1].set_ylabel("earthquakes / week")
    ax[1].set_title(
        "Axial Seamount seismicity (Wilcock/Zhang catalog)", fontsize=10, loc="left"
    )
    ax[1].legend(fontsize=7.5, ncol=2, frameon=False)
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Scaleworm abundance vs Axial seismicity, 2017–2024", fontsize=12)
    _finish(
        fig,
        ax[1],
        "figure_worms_vs_seismicity.png",
        f"Seismicity: Axial cabled-array catalog (Mc={mc}); Navy-diversion "
        "gaps affect raw counts. Cite Wilcock et al. 2016 Science 354:1395; "
        "Wilcock/Waldhauser/Tolstoy 2017 IEDA doi:10.1594/IEDA/323843.",
    )

    # fig 3: inflation
    infl = _geo("axial_inflation_weekly_2017_2024.csv")
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    _plot_worms(ax[0])
    ax[1].plot(infl.index, infl["uplift_cm"], color=GREEN, lw=1.5)
    ax[1].axhline(0, color="0.7", lw=0.6)
    ax[1].set_ylabel("uplift since 2017 (cm)")
    ax[1].set_title(
        "Caldera magma inflation (BOTPTA301 BOTSFLU uplift)", fontsize=10, loc="left"
    )
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle(
        "Scaleworm abundance vs caldera magma inflation, 2017–2024", fontsize=12
    )
    _finish(
        fig,
        ax[1],
        "figure_worms_vs_inflation.png",
        "Inflation: OOI BOTPTA301 BOTSFLU de-tided seafloor uplift "
        "(positive = magma recharge; Central Caldera ~2 km from ASHES).",
    )


if __name__ == "__main__":
    main()
