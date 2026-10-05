"""Manual vs automated dw/dt over the clear window — does the automation track the truth?

Overlays the MANUAL Monday series (box-corrected ground truth) and the AUTOMATED M-F harvest
(scene-sort-free, count_corrected) on the one window where both exist and the detector is
validated: the clear window 2021-09-01 .. 2023-08-11.

Panel A: monthly abundance, manual vs automated (shows the ~25-30% level offset from density
         compression — automated reads low on crowded peaks).
Panel B: dw/dt, manual vs automated (the validation that matters — does the automated rate of
         change agree in SIGN and TIMING with the manual one?). Pearson r on the matched months
         with a bootstrap 95% CI quantifies the shape agreement; amplitude is expected to be
         compressed, so r (shape/timing), not slope, is the fair metric.

Weekly mean = independent unit; monthly mean over weekly means; dw/dt gap-aware month-to-month.
matplotlib, Okabe-Ito, 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MANUAL = REPO / "validation/monday_manual_series/monday_manual_timeseries.csv"
HARVEST = REPO / "notebooks/harvest_mf_2021_2023.csv"
OUT = REPO / "notebooks/figure_dwdt_manual_vs_automated.png"

MAN_C = "#000000"  # manual ground truth (black)
AUT_C = "#0072B2"  # automated detector (blue)
SEED, N_BOOT = 20261004, 10000
CLEAR0, CLEAR1 = pd.Timestamp("2021-09-01"), pd.Timestamp("2023-08-11")


def weekly_mean(dates: pd.Series, vals: pd.Series) -> pd.Series:
    df = pd.DataFrame(
        {"w": pd.to_datetime(dates).dt.to_period("W-MON").dt.start_time, "v": vals}
    )
    return df.groupby("w")["v"].mean()


def monthly_mean(weekly: pd.Series) -> pd.Series:
    df = pd.DataFrame(
        {"m": weekly.index.to_period("M").to_timestamp(), "v": weekly.values}
    )
    return df.groupby("m")["v"].mean()


def gap_aware_dwdt(monthly: pd.Series) -> pd.Series:
    """Month-to-month change, consecutive calendar months only; indexed by the later month."""
    months, vals, idx = monthly.index, [], []
    for i in range(1, len(months)):
        gap = (months[i].year - months[i - 1].year) * 12 + (
            months[i].month - months[i - 1].month
        )
        if gap == 1:
            vals.append(monthly.iloc[i] - monthly.iloc[i - 1])
            idx.append(months[i])
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def boot_r_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    rs = []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        if np.std(x[s]) > 0 and np.std(y[s]) > 0:
            rs.append(np.corrcoef(x[s], y[s])[0, 1])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return r, float(lo), float(hi)


def main() -> None:
    m = pd.read_csv(MANUAL)
    m["dt"] = pd.to_datetime(m["date"])
    m = m[(m["dt"] >= CLEAR0) & (m["dt"] < CLEAR1)]
    man_m = monthly_mean(weekly_mean(m["dt"], m["mean_worms"]))
    man_d = gap_aware_dwdt(man_m)

    h = pd.read_csv(HARVEST)
    h["dt"] = pd.to_datetime(h["datetime_utc"]).dt.tz_localize(None)
    h = h[(h["count_raw"] > 0) & (h["dt"] >= CLEAR0) & (h["dt"] < CLEAR1)].copy()
    aut_m = monthly_mean(weekly_mean(h["dt"], h["count_corrected"]))
    aut_d = gap_aware_dwdt(aut_m)

    # matched dw/dt months for the agreement stat
    both = man_d.index.intersection(aut_d.index)
    r, lo, hi = boot_r_ci(man_d[both].to_numpy(), aut_d[both].to_numpy())

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)

    # A: abundance
    axA.plot(
        man_m.index,
        man_m.values,
        "o-",
        color=MAN_C,
        lw=2.2,
        ms=5,
        label="manual Monday (ground truth)",
    )
    axA.plot(
        aut_m.index,
        aut_m.values,
        "s-",
        color=AUT_C,
        lw=2.2,
        ms=5,
        label="automated M-F (detector)",
    )
    axA.set_ylabel("Scale-worms\nper Scene-1 frame", fontweight="bold", fontsize=11)
    axA.set_title(
        "A · Monthly abundance — automated reads ~25–30% low on crowded peaks "
        "(density compression)",
        fontsize=12,
        fontweight="bold",
    )
    axA.legend(framealpha=0.9, fontsize=9, loc="upper right")
    axA.set_ylim(0, None)
    axA.grid(alpha=0.3)

    # B: dw/dt
    axB.axhline(0, color="0.4", lw=1)
    axB.plot(
        man_d.index, man_d.values, "o-", color=MAN_C, lw=2.2, ms=5, label="manual dw/dt"
    )
    axB.plot(
        aut_d.index,
        aut_d.values,
        "s-",
        color=AUT_C,
        lw=2.2,
        ms=5,
        label="automated dw/dt",
    )
    axB.set_ylabel("dw/dt\n(worms · frame⁻¹ · month⁻¹)", fontweight="bold", fontsize=11)
    axB.set_title(
        "B · Rate of change — automated vs manual (sign & timing agreement)",
        fontsize=12,
        fontweight="bold",
    )
    axB.set_xlabel(
        "Date (monthly, clear window 2021-09 … 2023-08)", fontweight="bold", fontsize=11
    )
    axB.legend(framealpha=0.9, fontsize=9, loc="upper right")
    axB.grid(alpha=0.3)
    axB.text(
        0.015,
        0.03,
        f"Pearson r = {r:.2f}  (95% CI [{lo:.2f}, {hi:.2f}], n={len(both)} months)\n"
        "shape/timing agreement; amplitude compressed by density under-count",
        transform=axB.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        style="italic",
        color="0.25",
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )

    for ax in (axA, axB):
        for s in ax.spines.values():
            s.set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"matched dw/dt months: {len(both)}  Pearson r={r:.3f} [{lo:.3f},{hi:.3f}]")
    print(
        f"manual abundance {man_m.min():.1f}-{man_m.max():.1f}/fr | "
        f"automated {aut_m.min():.1f}-{aut_m.max():.1f}/fr"
    )
    sign_agree = int((np.sign(man_d[both]) == np.sign(aut_d[both])).sum())
    print(f"dw/dt sign agreement: {sign_agree}/{len(both)} months")


if __name__ == "__main__":
    main()
