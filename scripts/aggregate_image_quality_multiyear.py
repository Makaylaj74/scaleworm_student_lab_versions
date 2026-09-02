"""Aggregate the multi-year CAMHDA301 sharpness survey and rank clearest dates.

Reads image_quality_multiyear_2015_2026.csv, rolls up to month and camera-year
(HD camera is swapped ~early August each year, so camera-year = Aug(Y)->Aug(Y+1)
= unit CAMHDA301-Y), validates the automated index against the hand-sorted
2023-sharp/2024-blurry contrast, writes a monthly summary CSV, and an interactive
Plotly figure. Exploratory -> Plotly + Okabe-Ito.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

REPO = Path("/home/jovyan/scaleworm-student-lab")
SURVEY = REPO / "notebooks/image_quality_multiyear_2015_2026.csv"
SCENE1 = REPO / "notebooks/image_quality_timeseries.csv"  # hand-sorted 2023-24
OUT_MONTHLY = REPO / "notebooks/image_quality_multiyear_monthly.csv"
OUT_FIG = REPO / "notebooks/figure_sharpness_2015_2026.html"

# approx annual HD-camera swap (RCA/VISIONS cruise, late-Jul/early-Aug)
SWAPS = {y: f"{y}-08-01" for y in range(2015, 2027)}
OKABE = {"even": "#0072B2", "odd": "#D55E00"}  # alternate camera-years


def camera_unit(ts: pd.Timestamp) -> int:
    """Unit install year: after ~Aug swap -> this year, else previous year."""
    return ts.year if ts.month >= 8 else ts.year - 1


def main() -> None:
    raw = pd.read_csv(SURVEY)
    raw["dt"] = pd.to_datetime(raw["ts"], format="%Y%m%dT%H%M%S")
    raw["yr"] = raw["dt"].dt.year

    # ---- data-quality report (before filtering) ----
    print("=" * 70)
    print("DATA QUALITY per calendar year (sampled .mp4 recordings)")
    print(f"{'yr':>4} {'sampled':>7} {'usable':>6} {'corrupt':>7} {'all-black':>9}")
    for yr, g in raw.groupby("yr"):
        usable = (g["n_frames"] >= 4).sum()
        corrupt = (
            g["note"].eq("corrupt/no-duration").sum()
            + g["note"].eq("extract-failed").sum()
        )
        black = g["note"].eq("all-black").sum()
        print(f"{yr:>4} {len(g):>7} {usable:>6} {corrupt:>7} {black:>9}")

    # require >=4 non-black frames for a recording to enter the index
    df = raw[raw["n_frames"] >= 4].copy()
    df["month"] = df["dt"].dt.to_period("M").dt.to_timestamp()
    df["cam_unit"] = df["dt"].apply(camera_unit)

    # ---- monthly rollup (median of per-clip median sharpness) ----
    monthly = (
        df.groupby("month")
        .agg(
            n=("sharp_median", "size"),
            sharp_med=("sharp_median", "median"),
            sharp_q25=("sharp_median", lambda s: s.quantile(0.25)),
            sharp_q75=("sharp_median", lambda s: s.quantile(0.75)),
            brightness=("brightness", "median"),
            contrast=("contrast", "median"),
        )
        .reset_index()
    )
    monthly.to_csv(OUT_MONTHLY, index=False)

    # ---- validation vs hand-sorted scene-1 series ----
    print("=" * 70)
    print("VALIDATION: does this automated index reproduce 2023>2024?")
    for yr in (2023, 2024):
        m = df[df["dt"].dt.year == yr]["sharp_median"]
        print(f"  calendar {yr}: median sharpness = {m.median():.1f}  (n={len(m)})")
    if SCENE1.exists():
        s1 = pd.read_csv(SCENE1)
        s1["dt"] = pd.to_datetime(s1["dt"])
        for yr in (2023, 2024):
            m = s1[(s1["dt"].dt.year == yr) & s1["sharpness"].notna()]["sharpness"]
            print(f"    [hand-sorted scene-1 {yr}: median = {m.median():.1f}]")

    # ---- camera-year ranking ----
    cam = (
        df.groupby("cam_unit")
        .agg(
            n=("sharp_median", "size"),
            sharp_med=("sharp_median", "median"),
            months=("month", "nunique"),
        )
        .reset_index()
        .sort_values("sharp_med", ascending=False)
    )
    print("\n" + "=" * 70)
    print("CAMERA-YEARS RANKED BY MEDIAN SHARPNESS (clearest first)")
    print("  CAMHDA301-YYYY = installed ~Aug YYYY, runs to ~Aug YYYY+1")
    for _, r in cam.iterrows():
        print(
            f"  CAMHDA301-{int(r.cam_unit)}: median {r.sharp_med:6.1f}"
            f"   n={int(r.n):3d}  months={int(r.months)}"
        )

    # ---- clearest individual months ----
    print("\n" + "=" * 70)
    print("TOP 15 CLEAREST MONTHS (>=8 recordings sampled, to avoid tiny-n flukes)")
    top = monthly[monthly["n"] >= 8].sort_values("sharp_med", ascending=False).head(15)
    for _, r in top.iterrows():
        print(f"  {r['month']:%Y-%m}: median {r['sharp_med']:6.1f}  (n={int(r['n'])})")

    # ---- clearest individual recordings ----
    print("\n" + "=" * 70)
    print("TOP 15 CLEAREST INDIVIDUAL RECORDINGS")
    for _, r in df.sort_values("sharp_median", ascending=False).head(15).iterrows():
        print(
            f"  {r['dt']:%Y-%m-%d %H:%M}  sharp_med={r['sharp_median']:7.1f}  "
            f"{Path(r['path']).name}"
        )

    # ---- figure ----
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(monthly["month"]) + list(monthly["month"][::-1]),
            y=list(monthly["sharp_q75"]) + list(monthly["sharp_q25"][::-1]),
            fill="toself",
            fillcolor="rgba(0,114,178,0.12)",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            name="IQR",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["sharp_med"],
            mode="lines+markers",
            line={"color": OKABE["even"], "width": 2},
            marker={"size": 5},
            name="monthly median sharpness",
            hovertemplate="%{x|%Y-%m}<br>median=%{y:.0f}<extra></extra>",
        )
    )
    for d in SWAPS.values():
        fig.add_vline(x=d, line={"color": "grey", "width": 1, "dash": "dot"})
    fig.add_annotation(
        x=SWAPS[2016],
        y=1,
        yref="paper",
        yanchor="bottom",
        text="dotted = annual HD-camera swap (~Aug)",
        showarrow=False,
        font={"size": 11, "color": "grey"},
    )
    fig.update_layout(
        title="CAMHDA301 (Mushroom vent) image sharpness, 2015-2026 — relative index",
        xaxis_title="date",
        yaxis_title="Laplacian-variance sharpness (median of sample)",
        template="simple_white",
        width=1100,
        height=500,
    )
    fig.write_html(OUT_FIG)
    print(f"\nmonthly CSV -> {OUT_MONTHLY}\nfigure -> {OUT_FIG}")


if __name__ == "__main__":
    main()
