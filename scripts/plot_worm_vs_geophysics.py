"""Two worm-vs-geophysics figures from the completed 2021-2024 MANUAL series:

  fig 1: scaleworm abundance + Axial seismicity
  fig 2: scaleworm abundance + caldera magma inflation

Temperature is handled by ``compare_periods_worms_vs_temp.py`` (which also prints the
period-comparison statistics and owns the shared worm panel imported here). The worm
panel is now ground-truth manual box-counts (2021-2024), no longer the provisional v2
series. Geophysics is shown on the 2021-2024 worm-overlap window.

matplotlib 300 DPI, Okabe-Ito, gap-aware.

CAVEAT (stated in each caption): 2024 worm counts are blur-suppressed LOWER BOUNDS, so
the apparent worm decline while seismicity/inflation ramp is NOT a clean anti-correlation
and no correlation is asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from compare_periods_worms_vs_temp import (
    BLUR_ONSET,
    XLIM,
    load_ai2019,
    load_worms,
    plot_worm_panel,
)

NB = Path(__file__).resolve().parent.parent / "notebooks"
MC_TXT = NB / "axial_seismicity_weekly_2017_2024.mc.txt"

# Okabe-Ito
BLUE, BLACK, GREEN = "#0072B2", "#000000", "#009E73"


def _geo(name: str) -> pd.DataFrame:
    df = pd.read_csv(NB / name, parse_dates=["week_start"])
    df["week_start"] = df["week_start"].dt.tz_localize(None)
    # crop to the worm-overlap window so the y-axis is not squashed by 2017 extremes.
    return df[df["week_start"].between(*XLIM)]


def _finish(fig, ax_bottom, fname: str, cap: str) -> None:
    ax_bottom.axvline(BLUR_ONSET, color=BLACK, ls="--", lw=1.0, alpha=0.7)
    ax_bottom.grid(alpha=0.25, lw=0.5)
    ax_bottom.set_xlim(*XLIM)
    ax_bottom.xaxis.set_major_locator(mdates.MonthLocator((1, 7)))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_bottom.get_xticklabels(), rotation=45, ha="right")
    ax_bottom.set_xlabel("week (weekly Monday sampling)")
    full = (
        "AI-generated caption (Claude, Anthropic) — for review. Top: scale-worm (Polynoidae) "
        "abundance at the Mushroom vent, Axial Seamount (OOI CAMHDA301). SOLID blue = 2021-2024 "
        "manual box-counts (Monday mean ± SEM over ≤8 front-on Scene-1 slots, 128 Mondays "
        "2021-09..2024-12, 714 frames). OPEN orange (dashed) = 2019 AI-corrected index (v2 "
        "×2.24, gate recall 44.7%, 20 Mondays/102 frames), lower-confidence (bracket ×2.02–2.51). "
        "The 2019-07..2021-09 gap is empty by design (no footage / 2020 fails gate / spring-2021 "
        "unsorted; no interpolation). Dashed vertical = ~2023-08-10 camera-blur onset; post-onset "
        "manual counts are LOWER BOUNDS (reduced countability), so the apparent worm decline while "
        "the geophysics ramps is confounded and NO worm–geophysics correlation is asserted. "
        + cap
    )
    fig.text(
        0.01, 0.005, full, fontsize=6.4, color="#444", wrap=True, ha="left", va="bottom"
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    out = NB / fname
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    w = load_worms()
    mc = MC_TXT.read_text().strip() if MC_TXT.exists() else "?"

    # fig 1: seismicity
    seis = _geo("axial_seismicity_weekly_2017_2024.csv")
    fig, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True)
    plot_worm_panel(ax_w, w, ai=load_ai2019())
    ax_s.plot(
        seis["week_start"],
        seis["eq_count_all"],
        color=BLUE,
        lw=1.0,
        label="all located events",
    )
    ax_s.plot(
        seis["week_start"],
        seis["eq_count_mc"],
        color=BLACK,
        lw=0.8,
        ls=":",
        label=f"MW ≥ {mc} (Mc)",
    )
    ax_s.set_ylabel("earthquakes / week")
    ax_s.set_title(
        "Axial Seamount seismicity (Wilcock/Zhang cabled-array catalog)",
        fontsize=11,
        loc="left",
    )
    ax_s.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(
        "Scaleworm abundance vs Axial seismicity, 2019 (AI) + 2021–2024", fontsize=12
    )
    _finish(
        fig,
        ax_s,
        "figure_worms_vs_seismicity.png",
        f"Bottom: weekly located-earthquake counts (Mc={mc} by max-curvature); Navy "
        "high-rate-data diversions leave gaps that depress raw counts. Cite Wilcock et al. "
        "2016 Science 354:1395; Wilcock/Waldhauser/Tolstoy 2017 IEDA doi:10.1594/IEDA/323843.",
    )

    # fig 2: inflation
    infl = _geo("axial_inflation_weekly_2017_2024.csv")
    fig, (ax_w, ax_i) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True)
    plot_worm_panel(ax_w, w, ai=load_ai2019())
    ax_i.plot(infl["week_start"], infl["uplift_cm"], color=GREEN, lw=1.6)
    ax_i.set_ylabel("caldera uplift (cm)\nsince 2017")
    ax_i.set_title(
        "Caldera magma inflation (BOTPTA301 BOTSFLU de-tided uplift)",
        fontsize=11,
        loc="left",
    )
    fig.suptitle(
        "Scaleworm abundance vs caldera magma inflation, 2019 (AI) + 2021–2024",
        fontsize=12,
    )
    _finish(
        fig,
        ax_i,
        "figure_worms_vs_inflation.png",
        "Bottom: OOI BOTPTA301 BOTSFLU seafloor uplift (positive = magma recharge; Central "
        "Caldera ~2 km from ASHES), referenced to early-2017.",
    )


if __name__ == "__main__":
    main()
