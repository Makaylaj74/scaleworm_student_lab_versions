"""Static (matplotlib, 300 DPI) 3-panel version of the 2021-2024 weekly ASHES/
Axial geophysical signals — a publication/drop-in companion to the interactive
Plotly figure. Same data, same Okabe-Ito palette, gap-aware."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

NB = Path(__file__).resolve().parent.parent / "notebooks"
MC_TXT = NB / "axial_seismicity_weekly_2017_2024.mc.txt"
OUT_PNG = NB / "figure_geophysical_panels_2017_2024.png"

ORANGE, VERMILLION, BLUE, GREEN, BLACK = (
    "#E69F00",
    "#D55E00",
    "#0072B2",
    "#009E73",
    "#000000",
)


def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(NB / name, index_col="week_start", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    grid = pd.date_range("2017-01-02", "2024-12-30", freq="W-MON")
    return df.reindex(grid)


def main() -> None:
    therm = _read("ashes_thermistor_weekly_2017_2024.csv")
    seis = _read("axial_seismicity_weekly_2017_2024.csv")
    infl = _read("axial_inflation_weekly_2017_2024.csv")
    mc = MC_TXT.read_text().strip() if MC_TXT.exists() else "?"

    fig, ax = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)

    ax[0].plot(
        therm.index,
        therm["array_max"],
        color=VERMILLION,
        lw=1.3,
        label="hottest thermistor",
    )
    ax[0].plot(
        therm.index, therm["array_mean"], color=ORANGE, lw=1.2, label="array mean"
    )
    ax[0].set_ylabel("temperature (°C)")
    ax[0].set_title(
        "Mushroom diffuse-flow temperature (RS03ASHS-MJ03B-07-TMPSFA301)",
        fontsize=10,
        loc="left",
    )
    ax[0].legend(fontsize=8, ncol=2, frameon=False)

    ax[1].plot(
        seis.index, seis["eq_count_all"], color=BLUE, lw=1.2, label="all located events"
    )
    ax[1].plot(
        seis.index, seis["eq_count_mc"], color=BLACK, lw=0.9, ls=":", label=f"MW ≥ {mc}"
    )
    ax[1].set_ylabel("earthquakes / week")
    ax[1].set_title(
        "Axial Seamount seismicity (OOI cabled-array catalog — Wilcock & Zhang)",
        fontsize=10,
        loc="left",
    )
    ax[1].legend(fontsize=8, ncol=2, frameon=False)

    ax[2].plot(infl.index, infl["uplift_cm"], color=GREEN, lw=1.6)
    ax[2].axhline(0, color="0.7", lw=0.6)
    ax[2].set_ylabel("uplift since 2017 (cm)")
    ax[2].set_title(
        "Caldera magma inflation "
        "(RS03CCAL-MJ03F-05-BOTPTA301, BOTSFLU seafloor uplift)",
        fontsize=10,
        loc="left",
    )

    ax[2].set_xlim(pd.Timestamp("2017-01-01"), pd.Timestamp("2025-01-01"))
    ax[2].xaxis.set_major_locator(mdates.YearLocator())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax[2].set_xlabel("week (2017–2024)")
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)

    fig.suptitle(
        "ASHES / Axial Seamount geophysical signals vs scaleworm "
        "habitat, 2017–2024 (weekly)",
        fontsize=12,
    )
    caption = (
        "AI-generated caption (Claude, Anthropic) — Weekly means/counts on a "
        "common weekly-Monday grid; line breaks = data gaps. Temperature: OOI "
        "TMPSFA301 24-thermistor array (QARTOD-passed). Seismicity: Axial "
        f"cabled-array catalog (hypo71.dat, 2026-09-14; Mc={mc}); raw counts "
        "affected by Navy data-diversion gaps; cite Wilcock et al. 2016 Science "
        "354:1395 & Wilcock/Waldhauser/Tolstoy 2017 IEDA doi:10.1594/IEDA/"
        "323843. Inflation: OOI BOTPTA301 BOTSFLU de-tided seafloor uplift "
        "(positive = magma recharge; Central Caldera ~2 km from ASHES)."
    )
    fig.text(
        0.01,
        0.005,
        caption,
        fontsize=6.7,
        color="#444",
        wrap=True,
        ha="left",
        va="bottom",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=300)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
