"""Build the improved weekly clear-window abundance index from the v8+SAHI re-harvest and a
comparison figure vs the old v5 index and the manual Monday truth.

Index = weekly mean of count_v8sahi_max (max over a 60s grid, v8_tiled+SAHI) over the clear window
2021-09..2023-08. Validated at 100% of manual in aggregate (vs the old v5 index's 81%); residual is
density-dependent (sparse over / dense under) = the documented density ceiling.

Outputs:
  notebooks/v8_index_weekly_2021_2023.csv
  notebooks/figure_v8_index_vs_manual_2021_2023.png  (300 dpi)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path("/home/jovyan/scaleworm-student-lab")
LO, HI = "2021-09-01", "2023-08-11"
# Okabe-Ito (colorblind-safe)
C_V8, C_V5, C_MAN = "#0072B2", "#E69F00", "#000000"


def weekly(frame_ids, values):
    # coerce to plain lists so pd.Series does NOT realign a passed Series by its own index
    frame_ids = list(frame_ids)
    values = list(values)
    s = pd.Series(values, index=pd.to_datetime(
        [f"{f.split('-')[1][:8]}" for f in frame_ids], format="%Y%m%d"))
    s = s[(s.index >= LO) & (s.index < HI)]
    g = s.groupby(pd.Grouper(freq="W-MON"))
    return g.mean(), g.sem(), g.count()


def main() -> None:
    v8 = pd.read_csv(REPO / "notebooks/harvest_clear_v8sahi_2021_2023.csv")
    v5 = pd.read_csv(REPO / "notebooks/harvest_mf_2021_2023.csv")
    # manual Monday box counts
    man_ids, man_vals = [], []
    for lp in (REPO / "validation/monday_manual_series/labels").glob("CAMHDA301-*.txt"):
        d = lp.stem.split("-")[1][:8]
        if "20210901" <= d < "20230811":
            man_ids.append(lp.stem)
            man_vals.append(sum(1 for ln in lp.read_text().splitlines() if ln.strip()))

    v8m, v8e, v8n = weekly(v8["frame_id"], v8["count_v8sahi_max"])
    v5m, _, _ = weekly(v5["frame_id"], v5["count_corrected"])
    manm, _, _ = weekly(man_ids, man_vals)

    out = pd.DataFrame({"week": v8m.index, "v8sahi_mean": v8m.values,
                        "v8sahi_sem": v8e.values, "n_recordings": v8n.values})
    out.to_csv(REPO / "notebooks/v8_index_weekly_2021_2023.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    # reindex onto the v8 weekly axis so gaps stay as NaN (no silent interpolation)
    idx = v8m.index
    ax.fill_between(idx, (v8m - v8e).values, (v8m + v8e).values, color=C_V8, alpha=0.2, lw=0)
    ax.plot(idx, v8m.values, color=C_V8, lw=1.6, label="v8_tiled+SAHI index (weekly mean, max-over-grid)")
    ax.plot(v5m.reindex(idx).index, v5m.reindex(idx).values, color=C_V5, lw=1.4,
            ls="--", label="old v5 index (med-top3 ÷0.72)")
    ax.plot(manm.reindex(idx).index, manm.reindex(idx).values, "o", color=C_MAN,
            ms=4, label="manual Monday (box counts, truth)")

    pad = pd.Timedelta(days=int((idx[-1] - idx[0]).days * 0.03))
    ax.set_xlim(idx[0] - pad, idx[-1] + pad)
    ax.set_ylim(0, None)
    ax.set_xlabel("Date (ISO week, Mon)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Scale-worm abundance (worms / frame)", fontweight="bold", fontsize=11)
    ax.set_title("Automated scale-worm abundance index, clear window 2021-09–2023-08\n"
                 "CAMHDA301, ASHES Mushroom vent — v8+SAHI re-harvest vs v5 vs manual",
                 fontsize=12)
    ax.grid(alpha=0.3)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
    leg = ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    leg.get_frame().set_linewidth(0.8)

    cap = ("AI-generated caption (Claude, Anthropic) — Weekly-mean scale-worm abundance (worms per "
           "Scene-1 frame) on the OOI RS03ASHS CAMHDA301 camera, clear window 2021-09-01 to 2023-08-11 "
           "(blurry post-camera-swap era excluded). Blue: automated index from the v8_tiled detector with "
           "SAHI tiled inference, counting the max over a 60 s temporal grid per recording (all weekdays; "
           "derived file harvest_clear_v8sahi_2021_2023.csv). Orange dashed: prior v5 index (median-of-top-3 "
           "÷0.72). Black points: manual box counts on human-picked Monday Scene-1 frames "
           "(monday_manual_series). The v8+SAHI index matches the manual truth at 100% in aggregate "
           "(vs 81% for v5); residual is density-dependent (sparse frames over-counted via max-selection, "
           "dense peaks under-counted at the detection ceiling). Gaps = weeks without usable recordings "
           "(not interpolated).")
    fig.subplots_adjust(bottom=0.30, top=0.88)
    fig.text(0.5, 0.015, cap, ha="center", va="bottom", wrap=True, fontsize=7.5)
    fig.savefig(REPO / "notebooks/figure_v8_index_vs_manual_2021_2023.png", dpi=300,
                bbox_inches="tight")
    print("wrote notebooks/v8_index_weekly_2021_2023.csv")
    print("wrote notebooks/figure_v8_index_vs_manual_2021_2023.png")
    print(f"weeks={len(v8m.dropna())}  v8 mean={v8m.mean():.1f}  v5 mean={v5m.mean():.1f}  "
          f"manual mean={manm.mean():.1f}/fr")


if __name__ == "__main__":
    main()
