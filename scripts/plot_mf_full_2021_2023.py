"""Uniform all-automated Monday-Friday scaleworm abundance index, 2021-2023.

ONE method for every day: the scene-sort-free detector harvest (count_corrected = med-of-top-3
÷0.72, off-scene zeros dropped) applied uniformly to all M-F recordings. Manual Monday is NOT
merged in (that would mix a human scene-sort with the detector proxy) — it is overlaid only as
an INDEPENDENT weekly check. This is a validated RELATIVE INDEX, not a frame-verified absolute
count: the within-recording frame is the detector's max-worm pick (a Scene-1 proxy, ~90% right
in the clear window), ~9% of recordings are off-scene and dropped, so daily n varies (~7.4/8).
It tracks CHANGE; it is not "N worms". SAHI/density correction were dead ends, so no correction
beyond ÷0.72 (see results_note_dwdt_and_tierB / detector-v5 memory).

The CLEAR window (2021-09..2023-08) is validated (harvest = 90% of manual Monday) and drawn
solid; early-2021 (pre clear window) and blurry post-Aug-2023 are PROVISIONAL under-counts
(wrong recall regime) drawn dashed/greyed.

Writes the weekly regime-tagged series CSV + a 300-DPI figure. matplotlib, Okabe-Ito.
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
OUT_CSV = REPO / "notebooks/mf_index_weekly_2021_2023.csv"
CLEAR0, CLEAR1 = pd.Timestamp("2021-09-01"), pd.Timestamp("2023-08-11")
DET_C, MAN_C, PROV_C = "#0072B2", "#000000", "#999999"


def weekly(df, col):
    g = df.groupby(df.dt.dt.to_period("W-MON").dt.start_time)[col]
    return g.mean(), g.sem(), g.size()


def main() -> None:
    h = pd.read_csv(HARVEST)
    h["dt"] = pd.to_datetime(h.datetime_utc).dt.tz_localize(None)
    h = h[h.count_raw > 0].copy()  # drop off-scene "no usable view"
    clear = h[(h.dt >= CLEAR0) & (h.dt < CLEAR1)]
    early = h[h.dt < CLEAR0]  # pre clear window
    late = h[h.dt >= CLEAR1]  # blurry post-Aug-2023

    cw, cs, _ = weekly(clear, "count_corrected")
    ew, _, _ = weekly(early, "count_corrected")
    lw, _, _ = weekly(late, "count_corrected")

    # weekly regime-tagged series CSV (the data deliverable)
    def tag(df, label):
        mean, sem, n = weekly(df, "count_corrected")
        return pd.DataFrame({"week_start": mean.index, "mean_worms_per_frame": mean.values,
                             "sem": sem.values, "n_recordings": n.values, "regime": label})

    series = pd.concat([tag(early, "provisional_early2021"), tag(clear, "clear_validated"),
                        tag(late, "provisional_blurry2023")], ignore_index=True).sort_values("week_start")
    series.to_csv(OUT_CSV, index=False)

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
            label="manual Monday (independent check — not merged)")

    ax.set_xlabel("Date (weekly)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Scale-worms per Scene-1 frame\n(detector recall-corrected relative index)",
                  fontweight="bold", fontsize=12)
    ax.set_title("Uniform all-automated Monday-Friday scaleworm abundance index, 2021-2023\n"
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
    print(f"wrote {OUT_CSV}")
    print(f"clear weeks {len(cw)} (mean {cw.mean():.1f}/fr) | provisional weeks "
          f"{len(ew) + len(lw)} (early {len(ew)} + blurry {len(lw)}) "
          f"| total M-F recordings used {len(h)}")


if __name__ == "__main__":
    main()
