"""Interactive week-by-week explorer for the automated M-F abundance index, 2021-2023.

Exploratory / student-facing -> Plotly (hover, zoom, range slider). Lets you drill into the
accuracy of each count:
  - every ON-SCENE recording is a hoverable point: raw count, recall-corrected count, which
    second of the pan the detector picked (best_time_s), and the full per-frame scan vector
    (all_counts) so you can see whether the pick is stable or a single lucky frame;
  - the weekly automated mean (clear solid / provisional dashed) with +/-SEM band;
  - the MANUAL Monday truth overlaid -> the direct per-week accuracy check (where blue sits
    below black at peaks = the documented density compression, not a bug).

Writes a self-contained HTML (plotly.js embedded -> opens offline in JupyterLab / any browser).

Run: /home/jovyan/joseph-scaleworm-thesis/.venv/bin/python3 scripts/plot_mf_interactive.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parent.parent
HARVEST = REPO / "notebooks/harvest_mf_2021_2023.csv"
MON = REPO / "validation/monday_manual_series/monday_manual_timeseries.csv"
OUT = REPO / "notebooks/figure_mf_interactive_2021_2023.html"
CLEAR0, CLEAR1 = pd.Timestamp("2021-09-01"), pd.Timestamp("2023-08-11")

CLEAR_C, PROV_C, MAN_C = "#0072B2", "#999999", "#000000"


def regime_of(dt: pd.Series) -> pd.Series:
    r = pd.Series("clear_validated", index=dt.index)
    r[dt < CLEAR0] = "provisional_early2021"
    r[dt >= CLEAR1] = "provisional_blurry2023"
    return r


def weekly(df: pd.DataFrame):
    g = df.groupby(df["dt"].dt.to_period("W-MON").dt.start_time)["count_corrected"]
    return g.mean(), g.sem()


def main() -> None:
    h = pd.read_csv(HARVEST)
    h["dt"] = pd.to_datetime(h["datetime_utc"]).dt.tz_localize(None)
    h = h[h["count_raw"] > 0].copy()  # on-scene only (matches the index)
    h["regime"] = regime_of(h["dt"])

    fig = go.Figure()

    # per-recording points, by regime (drill-down layer)
    hovertmpl = (
        "<b>%{customdata[0]}</b><br>"
        "%{x|%Y-%m-%d %H:%M} (%{customdata[6]})<br>"
        "corrected: <b>%{y:.1f}</b> worms<br>"
        "raw: %{customdata[1]}  max: %{customdata[2]}<br>"
        "frame picked at: %{customdata[3]:.0f}s into pan<br>"
        "scan vector: %{customdata[4]}<extra></extra>"
    )
    for regime, color in [
        ("clear_validated", CLEAR_C),
        ("provisional_early2021", PROV_C),
        ("provisional_blurry2023", PROV_C),
    ]:
        s = h[h["regime"] == regime]
        if s.empty:
            continue
        cd = s[
            ["frame_id", "count_raw", "max_count", "best_time_s", "all_counts"]
        ].copy()
        cd["wd"] = s["dt"].dt.strftime("%a")
        cd.insert(5, "pad", "")  # keep index alignment simple for template [6]=wd
        fig.add_trace(
            go.Scattergl(
                x=s["dt"],
                y=s["count_corrected"],
                mode="markers",
                name=f"recording · {regime.split('_')[0]}",
                marker={"size": 4, "color": color, "opacity": 0.30},
                customdata=cd[
                    [
                        "frame_id",
                        "count_raw",
                        "max_count",
                        "best_time_s",
                        "all_counts",
                        "pad",
                        "wd",
                    ]
                ].to_numpy(),
                hovertemplate=hovertmpl,
                legendgroup="rec",
            )
        )

    # weekly means (clear solid + SEM band; provisional dashed)
    clear = h[h["regime"] == "clear_validated"]
    cw, cs = weekly(clear)
    fig.add_trace(
        go.Scatter(
            x=list(cw.index) + list(cw.index[::-1]),
            y=list((cw + cs).values) + list((cw - cs).values[::-1]),
            fill="toself",
            fillcolor="rgba(0,114,178,0.18)",
            line={"width": 0},
            name="±SEM",
            hoverinfo="skip",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cw.index,
            y=cw.values,
            mode="lines",
            line={"color": CLEAR_C, "width": 3},
            name="weekly mean (clear, validated)",
            hovertemplate="week of %{x|%Y-%m-%d}<br>mean <b>%{y:.1f}</b>/frame<extra></extra>",
        )
    )
    for seg in ("provisional_early2021", "provisional_blurry2023"):
        pw, _ = weekly(h[h["regime"] == seg])
        fig.add_trace(
            go.Scatter(
                x=pw.index,
                y=pw.values,
                mode="lines",
                line={"color": PROV_C, "width": 2, "dash": "dash"},
                name=f"weekly mean ({seg.split('_')[1]}, provisional)",
                hovertemplate="week of %{x|%Y-%m-%d}<br>mean %{y:.1f}/frame (provisional)<extra></extra>",
            )
        )

    # manual Monday truth (accuracy anchor)
    m = pd.read_csv(MON)
    m["dt"] = pd.to_datetime(m["date"])
    fig.add_trace(
        go.Scatter(
            x=m["dt"],
            y=m["mean_worms"],
            mode="lines+markers",
            line={"color": MAN_C, "width": 1.5, "dash": "dot"},
            marker={"size": 5, "color": MAN_C, "symbol": "diamond"},
            name="manual Monday (truth)",
            hovertemplate="manual %{x|%Y-%m-%d}<br><b>%{y:.1f}</b>/frame (hand-counted)<extra></extra>",
        )
    )

    fig.add_vrect(
        x0=h["dt"].min(), x1=CLEAR0, fillcolor=PROV_C, opacity=0.07, line_width=0
    )
    fig.add_vrect(
        x0=CLEAR1, x1=h["dt"].max(), fillcolor=PROV_C, opacity=0.07, line_width=0
    )

    fig.update_layout(
        title="Automated M-F scaleworm abundance — interactive week-by-week explorer (2021-2023)<br>"
        "<sub>hover any point for its raw/corrected count + which frame was picked; "
        "black = manual truth. Shaded = provisional regime.</sub>",
        xaxis={"title": "Date", "rangeslider": {"visible": True}},
        yaxis={
            "title": "Scale-worms per Scene-1 frame (recall-corrected index)",
            "rangemode": "tozero",
        },
        hovermode="closest",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        height=650,
    )
    fig.write_html(OUT, include_plotlyjs=True, full_html=True)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"on-scene recordings plotted: {len(h)}  | clear weeks {len(cw)}")


if __name__ == "__main__":
    main()
