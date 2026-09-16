"""Standalone 2019 figure: AI-corrected scale-worm abundance vs geophysics.

The 2019 (camera unit CAMHDA301-2018, Jan-Jul footage) worm counts are AI
detections (v2) recall-corrected ×2.24 from the per-unit gate (recall 44.7%,
which clears the 40% gate) — see build_gate_corrected_counts.py. This is a
lower-confidence index than the 2021-2024 manual box-counts and is drawn with
OPEN symbols + a dashed connector to signal that.

Four stacked panels over the 2019-01..2019-08 window (the only 2019 footage on
disk): worms, diffuse-flow temperature, Axial seismicity, caldera inflation.
No camera-blur marker (blur onset is 2023). matplotlib 300 DPI, Okabe-Ito.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

NB = Path(__file__).resolve().parent.parent / "notebooks"
AI = NB / "worm_timeseries_gate_corrected.csv"
SEIS = NB / "axial_seismicity_weekly_2017_2024.csv"
INFL = NB / "axial_inflation_weekly_2017_2024.csv"
THERM = NB / "ashes_thermistor_weekly_2017_2024.csv"

# window: the 2019 footage on disk (Jan-Jul); +2% pad each side
WIN = (pd.Timestamp("2018-12-25"), pd.Timestamp("2019-07-25"))

# Okabe-Ito
BLUE, ORANGE, VERMILLION, GREEN, BLACK = (
    "#0072B2",
    "#E69F00",
    "#D55E00",
    "#009E73",
    "#000000",
)


def _geo(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week_start"])
    df["week_start"] = df["week_start"].dt.tz_localize(None)
    return df[df["week_start"].between(*WIN)]


def _worms() -> pd.DataFrame:
    w = pd.read_csv(AI, parse_dates=["date"])
    w["sem"] = pd.to_numeric(w["sem_corrected"], errors="coerce")
    return w[w["date"].between(*WIN)].sort_values("date").reset_index(drop=True)


def main() -> None:
    w, seis, infl, therm = _worms(), _geo(SEIS), _geo(INFL), _geo(THERM)

    fig, axes = plt.subplots(4, 1, figsize=(9, 9.5), sharex=True)
    ax_w, ax_t, ax_s, ax_i = axes

    # (a) worms — AI-corrected, open symbols + dashed connector, gap-aware
    seg = w["date"].diff().dt.days.gt(14).cumsum()
    first = True
    for _, s in w.groupby(seg):
        ax_w.errorbar(
            s["date"],
            s["mean_corrected"],
            yerr=s["sem"],
            color=BLUE,
            marker="o",
            mfc="none",
            ms=4,
            lw=1.0,
            ls="--",
            elinewidth=0.6,
            capsize=1.5,
            label="AI-corrected (v2 ×2.24) mean ± SEM" if first else None,
        )
        first = False
    ax_w.set_ylabel("scale-worms / frame\n(AI-corrected)")
    ax_w.set_ylim(-3, ax_w.get_ylim()[1] * 1.18)  # headroom so tall point + legend don't collide
    ax_w.set_title(
        "AI-corrected scale-worm abundance — Mushroom vent, Axial Seamount (2019)",
        fontsize=11,
        loc="left",
    )
    ax_w.legend(fontsize=8, frameon=True, framealpha=0.9, loc="lower left")

    # (b) temperature
    ax_t.plot(
        therm["week_start"],
        therm["array_max"],
        color=VERMILLION,
        lw=1.2,
        label="hottest thermistor (localized)",
    )
    ax_t.plot(
        therm["week_start"],
        therm["array_mean"],
        color=ORANGE,
        lw=1.4,
        label="array mean (field)",
    )
    ax_t.set_ylabel("diffuse-flow\ntemperature (°C)")
    ax_t.set_title(
        "Mushroom diffuse-flow temperature — OOI TMPSFA301 array",
        fontsize=11,
        loc="left",
    )
    ax_t.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper right")

    # (c) seismicity
    ax_s.plot(seis["week_start"], seis["eq_count_all"], color=GREEN, lw=1.4)
    ax_s.set_ylabel("earthquakes\n/ week")
    ax_s.set_title(
        "Axial seismicity — UW cabled-array catalog (all events)",
        fontsize=11,
        loc="left",
    )

    # (d) inflation
    ax_i.plot(infl["week_start"], infl["uplift_cm"], color=BLACK, lw=1.4)
    ax_i.set_ylabel("uplift since\n2017 (cm)")
    ax_i.set_title(
        "Caldera inflation — OOI BOTPTA301 (Central Caldera)", fontsize=11, loc="left"
    )

    for lab, ax in zip("abcd", axes, strict=True):
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_xlim(*WIN)
        ax.text(
            -0.085,
            1.02,
            f"({lab})",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
    ax_i.xaxis.set_major_locator(mdates.MonthLocator())
    ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_i.get_xticklabels(), rotation=45, ha="right")
    ax_i.set_xlabel("week (weekly Monday sampling)")

    cap = (
        "AI-generated caption (Claude, Anthropic) — for review. (a) Scale-worm (Polynoidae) "
        "abundance at the Mushroom vent, Axial Seamount, from OOI HD video (CAMHDA301, camera "
        "unit 2018); counts are v2-detector boxes recall-corrected ×2.24 (gate recall 44.7% "
        "[39.9,49.4], clears the 40% gate) — a LOWER-CONFIDENCE index than manual box-counts, "
        "hence open symbols; error bars ±1 SEM over that Monday's Scene-1 slots. 20 Mondays, "
        "2019-01-07..2019-07-15, 102 frames. (b) OOI TMPSFA301 24-thermistor diffuse-flow array, "
        "QARTOD-passed weekly mean; 'hottest thermistor' is one localized sensor, not a field "
        "temperature. (c) Axial earthquake catalog (Wilcock/Waldhauser UW cabled array), weekly "
        "all-event count. (d) OOI BOTPTA301 Central-Caldera BOTSFLU seafloor uplift (cm since "
        "2017). All series weekly-Monday aggregated. No worm–geophysics correlation is asserted; "
        "the recall correction carries a ×2.02–2.51 bracket (recall CI). Derived from "
        "version-controlled scripts + data."
    )
    fig.text(
        0.01, 0.005, cap, fontsize=6.4, color="#444", wrap=True, ha="left", va="bottom"
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.99])
    out = NB / "figure_worms_2019_ai_vs_geophysics.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}  ({len(w)} worm Mondays, 2019 window)")


if __name__ == "__main__":
    main()
