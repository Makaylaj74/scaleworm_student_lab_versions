"""Side-by-side (stacked, shared time axis) 2021-2024 weekly panels of the three
ASHES/Axial geophysical signals for the scaleworm study:

  1. Mushroom diffuse-flow temperature  (RS03ASHS TMPSFA301 thermistor array)
  2. Axial Seamount seismicity          (Wilcock/Zhang cabled-array catalog)
  3. Caldera magma inflation            (RS03CCAL BOTPTA301 BOTSFLU uplift)

Reads the three weekly CSVs, reindexes onto a common weekly grid (gaps break
lines, no interpolation across them), writes an interactive Plotly HTML
(lab default for exploratory / student-facing work). Okabe-Ito colors; every
axis labelled with units; caption gives sources + required citations + caveats.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NB = Path(__file__).resolve().parent.parent / "notebooks"
THERM_CSV = NB / "ashes_thermistor_weekly_2017_2024.csv"
SEIS_CSV = NB / "axial_seismicity_weekly_2017_2024.csv"
INFL_CSV = NB / "axial_inflation_weekly_2017_2024.csv"
MC_TXT = NB / "axial_seismicity_weekly_2017_2024.mc.txt"
OUT_HTML = NB / "figure_geophysical_panels_2017_2024.html"

ORANGE = "#E69F00"
VERMILLION = "#D55E00"
BLUE = "#0072B2"
GREEN = "#009E73"
BLACK = "#000000"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="week_start", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def main() -> None:
    therm, seis, infl = _read(THERM_CSV), _read(SEIS_CSV), _read(INFL_CSV)
    mc = MC_TXT.read_text().strip() if MC_TXT.exists() else "?"

    grid = pd.date_range("2017-01-02", "2024-12-30", freq="W-MON")
    therm, seis, infl = (d.reindex(grid) for d in (therm, seis, infl))

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=(
            (
                "Mushroom diffuse-flow temperature "
                "(RS03ASHS-MJ03B-07-TMPSFA301, 24-thermistor array)"
            ),
            "Axial Seamount seismicity (OOI cabled-array catalog — Wilcock & Zhang)",
            (
                "Caldera magma inflation "
                "(RS03CCAL-MJ03F-05-BOTPTA301, BOTSFLU seafloor uplift)"
            ),
        ),
    )

    # 1) temperature
    fig.add_trace(
        go.Scatter(
            x=therm.index,
            y=therm["array_max"],
            name="hottest thermistor",
            mode="lines",
            line={"color": VERMILLION, "width": 1.6},
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d}<br>max %{y:.2f} °C<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=therm.index,
            y=therm["array_mean"],
            name="array mean",
            mode="lines",
            line={"color": ORANGE, "width": 1.4},
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d}<br>mean %{y:.2f} °C<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2) seismicity
    fig.add_trace(
        go.Scatter(
            x=seis.index,
            y=seis["eq_count_all"],
            name="all located events",
            mode="lines",
            line={"color": BLUE, "width": 1.4},
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f} eq/wk<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=seis.index,
            y=seis["eq_count_mc"],
            name=f"MW ≥ {mc}",
            mode="lines",
            line={"color": BLACK, "width": 1.0, "dash": "dot"},
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f} eq/wk<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # 3) inflation
    fig.add_trace(
        go.Scatter(
            x=infl.index,
            y=infl["uplift_cm"],
            name="cumulative uplift",
            mode="lines",
            line={"color": GREEN, "width": 1.8},
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} cm uplift<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.update_yaxes(title_text="temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="earthquakes / week", row=2, col=1)
    fig.update_yaxes(title_text="uplift since 2017 (cm)", row=3, col=1)
    fig.update_xaxes(
        title_text="week (2017–2024)", row=3, col=1, range=["2017-01-01", "2025-01-01"]
    )

    fig.update_layout(
        template="simple_white",
        height=980,
        width=1150,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "x": 0},
        margin={"t": 95, "b": 155},
        title={
            "text": "ASHES / Axial Seamount geophysical signals vs "
            "scaleworm habitat, 2017–2024 (weekly)",
            "x": 0.5,
            "xanchor": "center",
        },
    )
    caption = (
        "AI-generated caption (Claude, Anthropic) — Weekly means/counts, "
        "common weekly-Monday grid; line breaks = data gaps (no interpolation). "
        "TEMPERATURE: OOI RS03ASHS-MJ03B-07-TMPSFA301 24-thermistor diffuse-flow "
        "array at Mushroom, QARTOD-passed. SEISMICITY: Axial cabled-array "
        f"catalog (hypo71.dat, downloaded 2026-09-14; Mc={mc}, max-curvature); "
        "raw counts affected by Navy data-diversion gaps. Cite Wilcock et al. "
        "2016 Science 354:1395 & Wilcock, Waldhauser & Tolstoy 2017 IEDA "
        "doi:10.1594/IEDA/323843. INFLATION: OOI RS03CCAL-MJ03F-05-BOTPTA301 "
        "BOTSFLU daily de-tided seafloor depth; uplift = rise since the 2017 "
        "reference (positive = magma recharge). Central Caldera is ~2 km from "
        "ASHES. All OOI data first-party from the Hub archive."
    )
    fig.add_annotation(
        text=caption,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.15,
        showarrow=False,
        align="left",
        font={"size": 10, "color": "#444"},
        xanchor="left",
    )
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn")
    print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
