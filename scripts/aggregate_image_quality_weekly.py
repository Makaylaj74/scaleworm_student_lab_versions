"""Aggregate the weekly CLEAR-window survey (CAMHDA301 -2021/-2022 units,
Aug 2021 - Aug 2023) and pin the best stretches for scale-worm counting.

Rank by weekly MEDIAN sharpness (robust to marine-snow spikes). Reports top weeks,
monthly medians, and the best contiguous multi-week stretches; writes a Plotly
figure. Exploratory -> Plotly + Okabe-Ito.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

REPO = Path("/home/jovyan/scaleworm-student-lab")
SURVEY = REPO / "notebooks/image_quality_weekly_2021_2023.csv"
OUT_WEEKLY = REPO / "notebooks/image_quality_weekly_2021_2023_rollup.csv"
OUT_FIG = REPO / "notebooks/figure_sharpness_weekly_2021_2023.html"

SWAP_2022 = "2022-08-08"  # -2021 -> -2022 unit (RCA cruise TN407)
GOOD_CUT = 250.0  # ~crisp/countable (validated: ~300 crisp, ~135 blurry)


def main() -> None:
    raw = pd.read_csv(SURVEY)
    usable = raw[raw["n_frames"] >= 4].copy()
    usable["dt"] = pd.to_datetime(usable["ts"], format="%Y%m%dT%H%M%S")
    iso = usable["dt"].dt.isocalendar()
    usable["wk_start"] = [
        datetime.fromisocalendar(int(y), int(w), 1)
        for y, w in zip(iso["year"], iso["week"])
    ]

    wk = (
        usable.groupby("wk_start")
        .agg(
            n=("sharp_median", "size"),
            sharp_med=("sharp_median", "median"),
            q25=("sharp_median", lambda s: s.quantile(0.25)),
            q75=("sharp_median", lambda s: s.quantile(0.75)),
            black=("n_black", "sum"),
        )
        .reset_index()
        .sort_values("wk_start")
    )
    wk.to_csv(OUT_WEEKLY, index=False)

    print("=" * 66)
    print(f"CLEAR-WINDOW WEEKLY SURVEY  ({len(wk)} weeks, {len(usable)} recordings)")
    print(f"window median of weekly medians = {wk['sharp_med'].median():.0f}")

    print("\nTOP 15 CLEAREST WEEKS (>=4 recordings)")
    for _, r in (
        wk[wk["n"] >= 4].sort_values("sharp_med", ascending=False).head(15).iterrows()
    ):
        print(
            f"  week of {r['wk_start']:%Y-%m-%d}: median {r['sharp_med']:6.1f}  (n={int(r['n'])})"
        )

    # monthly medians across the window
    usable["month"] = usable["dt"].dt.to_period("M").dt.to_timestamp()
    mo = usable.groupby("month")["sharp_median"].median()
    print("\nMONTHLY MEDIAN across the window")
    for m, v in mo.items():
        bar = "#" * int(v / 12)
        print(f"  {m:%Y-%m}: {v:6.1f}  {bar}")

    # best contiguous stretch: longest run of consecutive weeks >= GOOD_CUT
    wk = wk.reset_index(drop=True)
    cur = None
    runs = []
    for i, r in wk.iterrows():
        if r["sharp_med"] >= GOOD_CUT:
            cur = cur or [i, i]
            cur[1] = i
        else:
            if cur:
                runs.append(tuple(cur))
                cur = None
    if cur:
        runs.append(tuple(cur))
    print(
        f"\nCONTIGUOUS STRETCHES with weekly median >= {GOOD_CUT:.0f} (crisp/countable):"
    )
    for a, b in sorted(runs, key=lambda x: x[1] - x[0], reverse=True)[:6]:
        wks = b - a + 1
        d0, d1 = wk.loc[a, "wk_start"], wk.loc[b, "wk_start"]
        med = wk.loc[a:b, "sharp_med"].median()
        print(f"  {d0:%Y-%m-%d} .. {d1:%Y-%m-%d}  ({wks} wk)  stretch median {med:.0f}")

    # ---- figure ----
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(wk["wk_start"]) + list(wk["wk_start"][::-1]),
            y=list(wk["q75"]) + list(wk["q25"][::-1]),
            fill="toself",
            fillcolor="rgba(0,114,178,0.12)",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wk["wk_start"],
            y=wk["sharp_med"],
            mode="lines+markers",
            line={"color": "#0072B2", "width": 2},
            marker={"size": 5},
            name="weekly median sharpness",
            hovertemplate="week of %{x|%Y-%m-%d}<br>median=%{y:.0f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=GOOD_CUT,
        line={"color": "#009E73", "width": 1, "dash": "dash"},
        annotation_text="crisp/countable (~250+)",
        annotation_position="top left",
    )
    fig.add_vline(x=SWAP_2022, line={"color": "#D55E00", "width": 1.5, "dash": "dot"})
    fig.add_annotation(
        x=SWAP_2022,
        y=1,
        yref="paper",
        yanchor="bottom",
        text="  -2021 unit | -2022 unit (Aug-2022 swap)",
        showarrow=False,
        font={"size": 11, "color": "#D55E00"},
    )
    fig.update_layout(
        title="CAMHDA301 clear window — weekly image sharpness, Aug 2021 - Aug 2023",
        xaxis_title="week",
        yaxis_title="Laplacian-variance sharpness (weekly median)",
        template="simple_white",
        width=1150,
        height=480,
    )
    fig.write_html(OUT_FIG)
    print(f"\nweekly rollup -> {OUT_WEEKLY}\nfigure -> {OUT_FIG}")


if __name__ == "__main__":
    main()
