"""Build the Tuesday-Friday weekday abundance series for the clear window (2021-09..2023-08)
from the v8+SAHI re-harvest, with Monday (automated + manual truth) as a validation overlay.

The v8+SAHI re-harvest (harvest_clear_v8sahi_2021_2023.csv) already counted every weekday; this
breaks out Tue-Fri as the automated whole-week series the advisor asked for (Monday was the only
hand-counted day; Tue-Fri are automation-only). Weekly means, clear window only.

Outputs:
  notebooks/tuefri_index_weekly_2021_2023.csv
  notebooks/figure_tuefri_series_2021_2023.png  (300 dpi)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path("/home/jovyan/scaleworm-student-lab")
LO, HI = "2021-09-01", "2023-08-11"
# Okabe-Ito
C_TF, C_MONA, C_MONM = "#009E73", "#56B4E9", "#000000"


def to_dt(fid):
    return pd.to_datetime(fid.split("-")[1][:8], format="%Y%m%d")


def weekly(df, mask):
    s = df[mask].set_index("dt")["count_v8sahi_max"]
    s = s[(s.index >= LO) & (s.index < HI)]
    g = s.groupby(pd.Grouper(freq="W-MON"))
    return g.mean(), g.sem()


def main() -> None:
    df = pd.read_csv(REPO / "notebooks/harvest_clear_v8sahi_2021_2023.csv")
    df["dt"] = df["frame_id"].map(to_dt)
    df["wd"] = df["dt"].dt.weekday  # Mon=0 .. Fri=4

    tf_m, tf_e = weekly(df, df["wd"].isin([1, 2, 3, 4]))     # Tue-Fri
    mon_m, _ = weekly(df, df["wd"] == 0)                       # Monday automated

    # manual Monday box-count truth
    man_ids, man_vals = [], []
    for lp in (REPO / "validation/monday_manual_series/labels").glob("CAMHDA301-*.txt"):
        d = lp.stem.split("-")[1][:8]
        if "20210901" <= d < "20230811":
            man_ids.append(lp.stem)
            man_vals.append(sum(1 for ln in lp.read_text().splitlines() if ln.strip()))
    man = pd.Series(man_vals, index=pd.to_datetime([i.split("-")[1][:8] for i in man_ids],
                                                   format="%Y%m%d")).groupby(
        pd.Grouper(freq="W-MON")).mean()

    out = pd.DataFrame({"week": tf_m.index, "tuefri_mean": tf_m.values,
                        "tuefri_sem": tf_e.values})
    out.to_csv(REPO / "notebooks/tuefri_index_weekly_2021_2023.csv", index=False)

    idx = tf_m.index
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.fill_between(idx, (tf_m - tf_e).values, (tf_m + tf_e).values, color=C_TF, alpha=0.2, lw=0)
    ax.plot(idx, tf_m.values, color=C_TF, lw=1.7, label="Tue-Fri automated index (v8+SAHI, weekly mean)")
    ax.plot(mon_m.reindex(idx).index, mon_m.reindex(idx).values, color=C_MONA, lw=1.2, ls="--",
            label="Monday automated (v8+SAHI)")
    ax.plot(man.reindex(idx).index, man.reindex(idx).values, "o", color=C_MONM, ms=3.5,
            label="Monday manual (box counts, truth)")

    pad = pd.Timedelta(days=int((idx[-1] - idx[0]).days * 0.03))
    ax.set_xlim(idx[0] - pad, idx[-1] + pad)
    ax.set_ylim(0, None)
    ax.set_xlabel("Date (ISO week, Mon)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Scale-worm abundance (worms / frame)", fontweight="bold", fontsize=11)
    ax.set_title("Tuesday-Friday automated scale-worm abundance, clear window 2021-09–2023-08\n"
                 "CAMHDA301, ASHES Mushroom vent (v8+SAHI), Monday as validation",
                 fontsize=12)
    ax.grid(alpha=0.3)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9).get_frame().set_linewidth(0.8)

    cap = ("AI-generated caption (Claude, Anthropic) — Weekly-mean scale-worm abundance (worms per "
           "Scene-1 frame) on OOI RS03ASHS CAMHDA301, clear window 2021-09-01 to 2023-08-11. Green: "
           "Tuesday-Friday automated index (v8_tiled detector + SAHI, max over a 60 s grid; the "
           "automation-only weekday series, derived file harvest_clear_v8sahi_2021_2023.csv). Blue dashed: "
           "Monday automated by the same method. Black points: manual box counts on human-picked Monday "
           "Scene-1 frames (monday_manual_series) — the only hand-counted day, shown as validation. The "
           "automated Tue-Fri series extends the weekly abundance record from 1 to ~4 counted days/week; "
           "it is a conservative relative/aggregate index (sparse frames over-counted, dense peaks under-"
           "counted at the detection ceiling). Gaps = weeks without usable recordings (not interpolated).")
    fig.subplots_adjust(bottom=0.30, top=0.88)
    fig.text(0.5, 0.015, cap, ha="center", va="bottom", wrap=True, fontsize=7.5)
    fig.savefig(REPO / "notebooks/figure_tuefri_series_2021_2023.png", dpi=300, bbox_inches="tight")

    print("wrote notebooks/tuefri_index_weekly_2021_2023.csv")
    print("wrote notebooks/figure_tuefri_series_2021_2023.png")
    print(f"Tue-Fri weeks={len(tf_m.dropna())}  Tue-Fri mean={tf_m.mean():.1f}  "
          f"Mon auto mean={mon_m.mean():.1f}  Mon manual mean={man.mean():.1f}/fr")


if __name__ == "__main__":
    main()
