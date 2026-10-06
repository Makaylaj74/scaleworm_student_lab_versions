"""Pair the automated scale-worm abundance index with ASHES diffuse-flow temperature.

Worms: weekly mean of the v8+SAHI automated index (clear window, all weekdays, 2021-2023, from
harvest_clear_v8sahi_2021_2023.csv + harvest_janaug2021_v8sahi.csv). Diffuse flow: weekly mean of
the ASHES thermistor array (ashes_thermistor_weekly_2021_2024.csv, array_mean = diffuse venting
temperature proxy). Exploratory only -- no causal claim; the automated worm index is a conservative
relative/aggregate measure that under-reads dense peaks (density ceiling).

Outputs:
  notebooks/worms_vs_diffuseflow_weekly.csv
  notebooks/figure_worms_vs_diffuseflow.png  (300 dpi)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/home/jovyan/scaleworm-student-lab")
LO, HI = "2021-01-01", "2023-08-11"
C_WORM, C_TEMP = "#0072B2", "#D55E00"


def worm_weekly() -> pd.Series:
    frames = [pd.read_csv(REPO / "notebooks/harvest_clear_v8sahi_2021_2023.csv")]
    ext = REPO / "notebooks/harvest_janaug2021_v8sahi.csv"
    if ext.exists() and ext.stat().st_size > 0:
        frames.append(pd.read_csv(ext))
    df = pd.concat(frames, ignore_index=True)
    df["dt"] = pd.to_datetime(df["frame_id"].str.split("-").str[1].str[:8], format="%Y%m%d")
    df = df[(df.dt >= LO) & (df.dt < HI)]
    return df.set_index("dt")["count_v8sahi_max"].groupby(pd.Grouper(freq="W-MON")).mean()


def temp_weekly() -> pd.Series:
    t = pd.read_csv(REPO / "notebooks/ashes_thermistor_weekly_2021_2024.csv")
    t["week"] = pd.to_datetime(t["week_start"], utc=True).dt.tz_localize(None)
    s = t.set_index("week")["array_mean"]
    s = s[(s.index >= LO) & (s.index < HI)]
    return s.groupby(pd.Grouper(freq="W-MON")).mean()


def main() -> None:
    w, t = worm_weekly(), temp_weekly()
    df = pd.concat([w.rename("worms"), t.rename("temp")], axis=1)
    df.to_csv(REPO / "notebooks/worms_vs_diffuseflow_weekly.csv")
    both = df.dropna()
    r = np.corrcoef(both["worms"], both["temp"])[0, 1]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.7, 1]})

    # Panel A: dual-axis time series
    axA.plot(w.index, w.values, color=C_WORM, lw=1.6, label="worms (automated)")
    axA.set_xlabel("Date (ISO week, Mon)", fontweight="bold", fontsize=11)
    axA.set_ylabel("Scale-worm abundance (worms / frame)", color=C_WORM, fontweight="bold", fontsize=11)
    axA.tick_params(axis="y", labelcolor=C_WORM)
    axA.set_ylim(0, None)
    axT = axA.twinx()
    axT.plot(t.index, t.values, color=C_TEMP, lw=1.6, label="diffuse-flow temp")
    axT.set_ylabel("Diffuse-flow temperature (°C, array mean)", color=C_TEMP, fontweight="bold", fontsize=11)
    axT.tick_params(axis="y", labelcolor=C_TEMP)
    axA.set_title("A  Weekly abundance vs diffuse-flow temperature", fontsize=11, loc="left", fontweight="bold")
    axA.grid(alpha=0.3)
    for lab in axA.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
    lines = axA.get_lines() + axT.get_lines()
    axA.legend(lines, [ln.get_label() for ln in lines], loc="upper left", framealpha=0.9, fontsize=9)

    # Panel B: scatter
    axB.scatter(both["temp"], both["worms"], s=18, color=C_WORM, alpha=0.5, edgecolor="none")
    axB.set_xlabel("Diffuse-flow temperature (°C)", fontweight="bold", fontsize=11)
    axB.set_ylabel("Worm abundance (worms / frame)", fontweight="bold", fontsize=11)
    axB.set_title(f"B  Weekly pairing (n={len(both)})", fontsize=11, loc="left", fontweight="bold")
    axB.grid(alpha=0.3)
    axB.text(0.04, 0.96, f"Pearson r = {r:.2f}", transform=axB.transAxes, va="top", ha="left",
             fontsize=10, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"})

    for ax in (axA, axB, axT):
        for sp in ax.spines.values():
            sp.set_linewidth(1.0)

    fig.suptitle("Automated scale-worm abundance vs ASHES diffuse-flow temperature, 2021–2023",
                 fontsize=12.5, fontweight="bold")
    cap = ("AI-generated caption (Claude, Anthropic) — Weekly scale-worm abundance (v8_tiled + SAHI "
           "automated index, max over a 60 s grid, all weekdays; harvest_clear_v8sahi_2021_2023.csv + "
           "harvest_janaug2021_v8sahi.csv) paired with ASHES diffuse-flow temperature (weekly array-mean of "
           "the 24-thermistor array, ashes_thermistor_weekly_2021_2024.csv), OOI RS03ASHS CAMHDA301, clear "
           "window 2021-2023. (A) Dual-axis time series; (B) weekly scatter with Pearson r. EXPLORATORY: no "
           "causal relationship is asserted. The worm index is a conservative relative measure that under-"
           "reads dense peaks (density ceiling), so peak magnitudes are lower bounds; Jan–Aug 2021 is "
           "unvalidated. Gaps not interpolated.")
    fig.subplots_adjust(bottom=0.32, top=0.86, wspace=0.30)
    fig.text(0.5, 0.015, cap, ha="center", va="bottom", wrap=True, fontsize=7.5)
    fig.savefig(REPO / "notebooks/figure_worms_vs_diffuseflow.png", dpi=300, bbox_inches="tight")
    print(f"wrote figure_worms_vs_diffuseflow.png + worms_vs_diffuseflow_weekly.csv  "
          f"(n={len(both)} weeks, Pearson r={r:.2f})")


if __name__ == "__main__":
    main()
