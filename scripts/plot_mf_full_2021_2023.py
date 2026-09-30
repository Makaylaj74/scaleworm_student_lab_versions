"""Full 2021-2023 Monday-Friday scaleworm abundance (all-detector harvest).

Weekly M-F mean from the scene-sort-free harvest (count_corrected = med-of-top-3 ÷0.72,
off-scene zeros dropped). The CLEAR window (2021-09..2023-08, units -2021/-2022) is validated
(harvest = 90% of manual Monday); outside it — early-2021 (pre clear window) and blurry
post-Aug-2023 — the detector regime is NOT validated and counts are PROVISIONAL under-counts
(wrong recall factor), drawn dashed/greyed. Manual Monday overlaid in the clear window as the
independent check.

matplotlib, Okabe-Ito, 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
HARVEST = REPO / "notebooks/harvest_mf_2021_2023.csv"
MON = REPO / "validation/monday_manual_series/monday_manual_timeseries.csv"
OUT = REPO / "notebooks/figure_mf_full_2021_2023.png"
CLEAR0, CLEAR1 = pd.Timestamp("2021-09-01"), pd.Timestamp("2023-08-11")
DET_C, MAN_C, PROV_C = "#0072B2", "#000000", "#999999"


def weekly(df, col):
    g = df.groupby(df.dt.dt.to_period("W-MON").dt.start_time)[col]
    return g.mean(), g.sem()


def main() -> None:
    h = pd.read_csv(HARVEST)
    h["dt"] = pd.to_datetime(h.datetime_utc).dt.tz_localize(None)
    h = h[h.count_raw > 0].copy()  # drop off-scene "no usable view"
    clear = h[(h.dt >= CLEAR0) & (h.dt < CLEAR1)]
    early = h[h.dt < CLEAR0]  # pre clear window
    late = h[h.dt >= CLEAR1]  # blurry post-Aug-2023

    cw, cs = weekly(clear, "count_corrected")
    ew, _ = weekly(early, "count_corrected")
    lw, _ = weekly(late, "count_corrected")

    m = pd.read_csv(MON)
    m["dt"] = pd.to_datetime(m.date)
    m = m[(m.dt >= CLEAR0) & (m.dt < CLEAR1)]
    mon_w = m.set_index(m.dt.dt.to_period("W-MON").dt.start_time)["mean_worms"]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axvspan(h.dt.min(), CLEAR0, color=PROV_C, alpha=0.10)
    ax.axvspan(CLEAR1, h.dt.max(), color=PROV_C, alpha=0.10)
    ax.text(CLEAR1, ax.get_ylim()[1], " blurry\n provisional", color=PROV_C, fontsize=8,
            va="top", fontweight="bold")

    ax.plot(cw.index, cw.values, "-", color=DET_C, lw=2.4, label="M-F detector (clear, validated)")
    ax.fill_between(cw.index, cw - cs, cw + cs, color=DET_C, alpha=0.2)
    ax.plot(ew.index, ew.values, "--", color=PROV_C, lw=1.6,
            label="M-F detector (provisional — regime not validated)")
    ax.plot(lw.index, lw.values, "--", color=PROV_C, lw=1.6)  # blurry segment, same style
    ax.plot(mon_w.index, mon_w.values, ":", color=MAN_C, lw=1.5, alpha=0.8,
            label="manual Monday (clear-window check)")

    ax.set_xlabel("Date (weekly)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Scale-worms per Scene-1 frame\n(detector recall-corrected)",
                  fontweight="bold", fontsize=12)
    ax.set_title("Automated Monday-Friday scaleworm abundance, 2021-2023\n"
                 "Clear window validated (90% of manual); early-2021 & blurry-2023 provisional",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, None)
    span = (h.dt.max() - h.dt.min()).days
    ax.set_xlim(h.dt.min() - pd.Timedelta(days=span * 0.02), h.dt.max() + pd.Timedelta(days=span * 0.02))
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.9, fontsize=9, loc="upper left")
    for s in ax.spines.values():
        s.set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"clear weeks {len(cw)} (mean {cw.mean():.1f}/fr) | provisional weeks "
          f"{len(ew) + len(lw)} (early {len(ew)} + blurry {len(lw)}) "
          f"| total M-F recordings used {len(h)}")


if __name__ == "__main__":
    main()
