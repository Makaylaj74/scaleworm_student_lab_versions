"""Two figures from the v8+SAHI re-harvest:
  (1) figure_mf_automated_2021_2023.png  -- automated Monday-Friday weekly abundance.
  (2) figure_monday_manual_vs_auto.png   -- manual vs automated Monday (time series + scatter).

Reads notebooks/harvest_clear_v8sahi_2021_2023.csv (+ optional janaug2021 extension if present) and
the manual Monday box counts. Clear window only. Weekly means.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/home/jovyan/scaleworm-student-lab")
LO, HI = "2021-01-01", "2023-08-11"
C_AUTO, C_MON, C_MANUAL = "#0072B2", "#56B4E9", "#000000"


def load_harvest() -> pd.DataFrame:
    frames = [pd.read_csv(REPO / "notebooks/harvest_clear_v8sahi_2021_2023.csv")]
    ext = REPO / "notebooks/harvest_janaug2021_v8sahi.csv"
    # only extend once the Jan-Aug 2021 harvest is near-complete (~1365 recs), not mid-run
    if ext.exists() and ext.stat().st_size > 0:
        e = pd.read_csv(ext)
        if len(e) >= 1300:
            frames.append(e)
        else:
            print(f"  (skipping partial Jan-Aug 2021 extension: {len(e)}/1365 recs)")
    df = pd.concat(frames, ignore_index=True)
    df["dt"] = pd.to_datetime(df["frame_id"].str.split("-").str[1].str[:8], format="%Y%m%d")
    df["wd"] = df["dt"].dt.weekday
    return df[(df["dt"] >= LO) & (df["dt"] < HI)]


def manual_monday() -> pd.Series:
    ids, vals = [], []
    for lp in (REPO / "validation/monday_manual_series/labels").glob("CAMHDA301-*.txt"):
        d = lp.stem.split("-")[1][:8]
        if "20210101" <= d < "20230811":
            ids.append(lp.stem)
            vals.append(sum(1 for ln in lp.read_text().splitlines() if ln.strip()))
    return pd.Series(vals, index=pd.Index(ids, name="frame_id"))


def weekly(dt_index, values):
    s = pd.Series(list(values), index=pd.DatetimeIndex(list(dt_index)))
    g = s.groupby(pd.Grouper(freq="W-MON"))
    return g.mean(), g.sem()


def fig_mf(df: pd.DataFrame) -> None:
    m, e = weekly(df["dt"], df["count_v8sahi_max"])
    idx = m.index
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.fill_between(idx, (m - e).values, (m + e).values, color=C_AUTO, alpha=0.2, lw=0)
    ax.plot(idx, m.values, color=C_AUTO, lw=1.7, label="Automated M–F index (v8+SAHI, weekly mean ± SEM)")
    pad = pd.Timedelta(days=int((idx[-1] - idx[0]).days * 0.03))
    ax.set_xlim(idx[0] - pad, idx[-1] + pad)
    ax.set_ylim(0, None)
    ax.set_xlabel("Date (ISO week, Mon)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Scale-worm abundance (worms / frame)", fontweight="bold", fontsize=11)
    ax.set_title("Automated Monday–Friday scale-worm abundance, 2021–2023\n"
                 "CAMHDA301, ASHES Mushroom vent (v8_tiled + SAHI)", fontsize=12)
    ax.grid(alpha=0.3)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9).get_frame().set_linewidth(0.8)
    span = f"{idx.min().date()} to {idx.max().date()}"
    cap = ("AI-generated caption (Claude, Anthropic) — Weekly-mean scale-worm abundance (worms per "
           f"Scene-1 frame) on OOI RS03ASHS CAMHDA301, all weekdays Mon–Fri, {span} (clear window; "
           "blurry post-2023-08 era excluded). Automated by the v8_tiled detector with SAHI tiled "
           "inference, max over a 60 s grid per recording (derived files harvest_clear_v8sahi_2021_2023.csv "
           "and harvest_janaug2021_v8sahi.csv). Conservative relative/aggregate index (sparse frames "
           "over-counted via max-selection, dense peaks under-counted at the detection ceiling). "
           "NOTE: manual box-count validation exists only from 2021-09 onward; the Jan–Aug 2021 segment is "
           "unvalidated (no manual truth, and the pre-Sep-2021 frame-finder was unreliable under the prior "
           "detector) — Mar–Dec 2021 is continuous with the validated window but Jan–Feb reads low (genuine "
           "early-year low or residual artifact, unresolved). Shaded band = ±1 SEM across the ~20–40 "
           "recordings per week; gaps = weeks without usable recordings (not interpolated).")
    fig.subplots_adjust(bottom=0.28, top=0.88)
    fig.text(0.5, 0.015, cap, ha="center", va="bottom", wrap=True, fontsize=7.5)
    fig.savefig(REPO / "notebooks/figure_mf_automated_2021_2023.png", dpi=300, bbox_inches="tight")
    print(f"wrote figure_mf_automated_2021_2023.png  (weeks={len(m.dropna())} span {span})")


def fig_monday(df: pd.DataFrame, man: pd.Series) -> None:
    mon = df[df["wd"] == 0].set_index("frame_id")
    # matched per-recording pairs for the scatter
    both = mon.join(man.rename("manual"), how="inner").dropna(subset=["manual"])
    am, _ = weekly(mon["dt"], mon["count_v8sahi_max"])
    man_dt = pd.to_datetime(man.index.str.split("-").str[1].str[:8], format="%Y%m%d")
    mm, _ = weekly(man_dt, man.values)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.6, 1]})

    # Panel A: time series
    axA.plot(am.index, am.values, color=C_MON, lw=1.5, label="Automated Monday (v8+SAHI)")
    axA.plot(mm.index, mm.values, "o", color=C_MANUAL, ms=4, label="Manual Monday (box counts)")
    pad = pd.Timedelta(days=int((am.index[-1] - am.index[0]).days * 0.03))
    axA.set_xlim(am.index[0] - pad, am.index[-1] + pad)
    axA.set_ylim(0, None)
    axA.set_xlabel("Date (ISO week, Mon)", fontweight="bold", fontsize=11)
    axA.set_ylabel("Scale-worm abundance (worms / frame)", fontweight="bold", fontsize=11)
    axA.set_title("A  Monday abundance over time", fontsize=11, loc="left", fontweight="bold")
    axA.grid(alpha=0.3)
    axA.legend(loc="upper left", framealpha=0.9, fontsize=9)
    for lab in axA.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")

    # Panel B: scatter manual vs automated (per recording)
    x = both["manual"].to_numpy(float)
    y = both["count_v8sahi_max"].to_numpy(float)
    r = np.corrcoef(x, y)[0, 1]
    ratio = y.sum() / x.sum()
    mae = np.abs(y - x).mean()
    lim = max(x.max(), y.max()) * 1.05
    axB.plot([0, lim], [0, lim], color="0.6", ls="--", lw=1, label="1:1")
    axB.scatter(x, y, s=16, color=C_MON, alpha=0.5, edgecolor="none")
    axB.set_xlim(0, lim)
    axB.set_ylim(0, lim)
    axB.set_aspect("equal")
    axB.set_xlabel("Manual count (worms / frame)", fontweight="bold", fontsize=11)
    axB.set_ylabel("Automated v8+SAHI (worms / frame)", fontweight="bold", fontsize=11)
    axB.set_title(f"B  Per-recording, n={len(both)}", fontsize=11, loc="left", fontweight="bold")
    axB.grid(alpha=0.3)
    axB.text(0.04, 0.96, f"r = {r:.2f}\nratio = {ratio:.0%}\nMAE = {mae:.1f}",
             transform=axB.transAxes, va="top", ha="left", fontsize=9,
             bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"})
    axB.legend(loc="lower right", framealpha=0.9, fontsize=9)
    for ax in (axA, axB):
        for sp in ax.spines.values():
            sp.set_linewidth(1.0)

    cap = ("AI-generated caption (Claude, Anthropic) — Manual vs automated Monday scale-worm abundance, "
           "OOI RS03ASHS CAMHDA301, clear window 2021-09 to 2023-08. (A) Weekly-mean automated Monday "
           "index (v8_tiled + SAHI, max over a 60 s grid) vs manual box counts on human-picked Monday "
           "Scene-1 frames. (B) Per-recording manual vs automated with the 1:1 line; Pearson r, count ratio "
           "(Σauto/Σmanual) and mean absolute error annotated. Points above 1:1 at low counts (max-selection "
           "over-count) and below 1:1 at high counts (density ceiling) show the density-dependent residual.")
    fig.subplots_adjust(bottom=0.34, top=0.90, wspace=0.28)
    fig.text(0.5, 0.015, cap, ha="center", va="bottom", wrap=True, fontsize=7.5)
    fig.savefig(REPO / "notebooks/figure_monday_manual_vs_auto.png", dpi=300, bbox_inches="tight")
    print(f"wrote figure_monday_manual_vs_auto.png  (n={len(both)} r={r:.2f} ratio={ratio:.0%} MAE={mae:.1f})")


def main() -> None:
    df = load_harvest()
    fig_mf(df)
    fig_monday(df, manual_monday())


if __name__ == "__main__":
    main()
