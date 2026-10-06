"""Count-dependent calibration for the dense-peak under-count -- a NEGATIVE RESULT: it does NOT work,
and this script documents why, rigorously.

Fits a MONOTONIC map automated_count -> manual_count (isotonic regression) on the 530 Monday
recordings with manual box-count truth, VALIDATES on a held-out 30% split. FINDING: the calibration
leaves the index essentially unchanged and does NOT lift the dense peaks (held-out dense ratio
82% -> 80%). ROOT CAUSE (demonstrated): the automated count is strongly MANY-TO-ONE -- for recordings
the detector counted at auto=18-22, the true manual count ranges from 1 to 43 (dense, moderate, and
false-positive-heavy frames all collapse to the same automated count). No function of the count alone
can separate them, so no count-based correction can recover the dense peaks -- the same reason the
earlier global power-law failed. Fixing the peaks needs MORE INFORMATION PER FRAME (a density-estimation
model that reads the image), or the limit is logged. Raw and 'calibrated' are both reported to show
they coincide.

Outputs:
  notebooks/v8_index_weekly_calibrated_2021_2023.csv
  notebooks/figure_index_raw_vs_calibrated.png  (300 dpi)
  validation/count_calibration.json  (the deployed map, for provenance)
Prints a held-out validation table (ratio / MAE, raw vs calibrated, by density) with bootstrap CIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

REPO = Path("/home/jovyan/scaleworm-student-lab")
LO, HI = "2021-09-01", "2023-08-11"
RNG = np.random.default_rng(20261006)
C_RAW, C_CAL, C_MAN = "#0072B2", "#D55E00", "#000000"  # Okabe-Ito


def monday_pairs():
    h = pd.read_csv(REPO / "notebooks/harvest_clear_v8sahi_2021_2023.csv").set_index("frame_id")
    man = {}
    for lp in (REPO / "validation/monday_manual_series/labels").glob("CAMHDA301-*.txt"):
        d = lp.stem.split("-")[1][:8]
        if "20210901" <= d < "20230811":
            man[lp.stem] = sum(1 for ln in lp.read_text().splitlines() if ln.strip())
    rows = [(fid, m, float(h.loc[fid, "count_v8sahi_max"]))
            for fid, m in man.items() if fid in h.index]
    return pd.DataFrame(rows, columns=["frame_id", "manual", "auto"])


def boot_ratio(auto, manual, n=5000):
    auto, manual = np.asarray(auto, float), np.asarray(manual, float)
    idx = RNG.integers(0, len(auto), size=(n, len(auto)))
    r = auto[idx].sum(1) / manual[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main() -> None:
    df = monday_pairs()
    # held-out validation (random 70/30, seeded)
    perm = RNG.permutation(len(df))
    cut = int(len(df) * 0.7)
    tr, te = df.iloc[perm[:cut]], df.iloc[perm[cut:]]
    iso = IsotonicRegression(out_of_bounds="clip").fit(tr["auto"], tr["manual"])

    te = te.assign(cal=iso.predict(te["auto"]))
    print(f"held-out validation (n={len(te)}):")
    for name, sub in (("ALL", te), ("sparse(man<=15)", te[te.manual <= 15]),
                      ("dense(man>15)", te[te.manual > 15])):
        rr = sub.auto.sum() / sub.manual.sum()
        cr = sub.cal.sum() / sub.manual.sum()
        rlo, rhi = boot_ratio(sub.auto, sub.manual)
        clo, chi = boot_ratio(sub.cal, sub.manual)
        print(f"  {name:16} n={len(sub):3d}  raw ratio {rr:5.0%} [{rlo:.0%},{rhi:.0%}]  "
              f"calib {cr:5.0%} [{clo:.0%},{chi:.0%}]  |  MAE raw {np.abs(sub.auto - sub.manual).mean():.1f} "
              f"-> calib {np.abs(sub.cal - sub.manual).mean():.1f}")

    # deploy: fit on ALL 530, apply to every weekday
    dep = IsotonicRegression(out_of_bounds="clip").fit(df["auto"], df["manual"])
    (REPO / "validation/count_calibration.json").write_text(json.dumps({
        "method": "isotonic automated->manual, fit on 530 Monday manual pairs",
        "x_thresholds": dep.X_thresholds_.tolist(), "y_thresholds": dep.y_thresholds_.tolist(),
    }, indent=1))

    h = pd.read_csv(REPO / "notebooks/harvest_clear_v8sahi_2021_2023.csv")
    h["dt"] = pd.to_datetime(h["frame_id"].str.split("-").str[1].str[:8], format="%Y%m%d")
    h = h[(h.dt >= LO) & (h.dt < HI)]
    h["raw"] = h["count_v8sahi_max"]
    h["cal"] = dep.predict(h["count_v8sahi_max"])

    def weekly(col):
        return h.set_index("dt")[col].groupby(pd.Grouper(freq="W-MON")).mean()

    wr, wc = weekly("raw"), weekly("cal")
    man = monday_pairs()
    man_dt = pd.to_datetime(man.frame_id.str.split("-").str[1].str[:8], format="%Y%m%d")
    wm = pd.Series(man.manual.values, index=man_dt).groupby(pd.Grouper(freq="W-MON")).mean()

    pd.DataFrame({"week": wr.index, "raw_mean": wr.values,
                  "calibrated_mean": wc.reindex(wr.index).values}).to_csv(
        REPO / "notebooks/v8_index_weekly_calibrated_2021_2023.csv", index=False)

    idx = wr.index
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(idx, wr.values, color=C_RAW, lw=1.3, alpha=0.8, label="Raw automated index (v8+SAHI)")
    ax.plot(idx, wc.reindex(idx).values, color=C_CAL, lw=1.8, label="Calibrated index (isotonic, held-out-validated)")
    ax.plot(wm.reindex(idx).index, wm.reindex(idx).values, "o", color=C_MAN, ms=3.5, label="Manual Monday (truth)")
    pad = pd.Timedelta(days=int((idx[-1] - idx[0]).days * 0.03))
    ax.set_xlim(idx[0] - pad, idx[-1] + pad)
    ax.set_ylim(0, None)
    ax.set_xlabel("Date (ISO week, Mon)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Scale-worm abundance (worms / frame)", fontweight="bold", fontsize=11)
    ax.set_title("Count-calibration does NOT fix the dense peaks (negative result)\n"
                 "CAMHDA301 clear window 2021-09–2023-08; raw and calibrated indices coincide", fontsize=12)
    ax.grid(alpha=0.3)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9).get_frame().set_linewidth(0.8)
    cap = ("AI-generated caption (Claude, Anthropic) — Weekly-mean scale-worm abundance (worms per Scene-1 "
           "frame), OOI RS03ASHS CAMHDA301, clear window. Blue: raw automated index (v8+SAHI, max over 60 s "
           "grid). Orange: same index after a monotonic count-dependent calibration (isotonic regression of "
           "automated onto manual, fit on 530 Monday manual pairs, validated on a 30% held-out split; "
           "validation/count_calibration.json). Black: manual Monday box counts (truth). NEGATIVE RESULT: "
           "the calibrated (orange) and raw (blue) curves coincide and the dense peaks are NOT lifted "
           "(held-out dense ratio 82%->80%), because the automated count is many-to-one — recordings with "
           "true manual counts of 1 to 43 all produce the same auto count, so no count-based map can recover "
           "the dense peaks. All weekdays Mon-Fri; gaps not interpolated.")
    fig.subplots_adjust(bottom=0.30, top=0.88)
    fig.text(0.5, 0.015, cap, ha="center", va="bottom", wrap=True, fontsize=7.5)
    fig.savefig(REPO / "notebooks/figure_index_raw_vs_calibrated.png", dpi=300, bbox_inches="tight")
    print("\nwrote v8_index_weekly_calibrated_2021_2023.csv + figure_index_raw_vs_calibrated.png")
    print(f"deployed map; weekly raw mean {wr.mean():.1f} -> calibrated {wc.mean():.1f} (manual {wm.mean():.1f})")


if __name__ == "__main__":
    main()
