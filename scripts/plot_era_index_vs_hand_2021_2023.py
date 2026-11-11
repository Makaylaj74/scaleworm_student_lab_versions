"""Automated era-time worm index vs hand counts, Jan 2021 - Aug 2023 (slide figure).

Weekly automated index (build_era_time_index_2021_2023.py: density head read at each
camera-schedule era's Scene-1 time, off-scene recordings < 8.5 dropped, day mean -> week
mean, Mon-Fri) against the weekly hand-count series under the counting rule (same source
as plot_worms_inflation_recount_2021_2023.py). Agreement statistics are computed here and
written into the caption: bootstrap over weeks, 10 000 resamples, seed 0, percentile CIs.

    .venv/bin/python3 scripts/plot_era_index_vs_hand_2021_2023.py
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_worms_inflation_recount_2021_2023 import END, NB, START, load, on_weeks

INDEX = NB / "era_time_index_2021_2023/era_time_index_weekly_2021_2023.csv"
CLEAR_END = pd.Timestamp("2023-08-10")
OUT = NB / "figure_era_index_vs_hand_2021_2023"

# Okabe-Ito
BLUE, ORANGE = "#0072B2", "#E69F00"
TITLE_PT, AXIS_PT, TICK_PT, LW = 22, 16, 14, 2.2


def agreement(auto: np.ndarray, hand: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Sum-ratio auto/hand and Pearson r with percentile bootstrap CIs over weeks."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hand), (10_000, len(hand)))
    ratio = auto[idx].sum(1) / hand[idx].sum(1)
    r = [np.corrcoef(auto[i], hand[i])[0, 1] for i in idx]
    return {
        "n": len(hand),
        "ratio": auto.sum() / hand.sum(),
        "ratio_lo": np.percentile(ratio, 2.5),
        "ratio_hi": np.percentile(ratio, 97.5),
        "r": np.corrcoef(auto, hand)[0, 1],
        "r_lo": np.percentile(r, 2.5),
        "r_hi": np.percentile(r, 97.5),
    }


def fmt(a: dict[str, float]) -> str:
    return (
        f"{a['ratio']:.2f}× [{a['ratio_lo']:.2f}–{a['ratio_hi']:.2f}], "
        f"r = {a['r']:.2f} [{a['r_lo']:.2f}–{a['r_hi']:.2f}], n = {a['n']} weeks"
    )


def main() -> None:
    hand = load()["rule"]
    auto = pd.read_csv(INDEX, parse_dates=["week"]).set_index("week")
    auto = auto[auto.index <= CLEAR_END]
    j = pd.concat([hand["mean"].rename("hand"), auto.worm_index.rename("auto")], axis=1)
    j = j.dropna()
    j = j[j.index <= CLEAR_END]
    all_ = agreement(j.auto.to_numpy(), j.hand.to_numpy())
    early = j[j.index < "2021-09-01"]
    late = j[j.index >= "2021-09-01"]
    a_early = agreement(early.auto.to_numpy(), early.hand.to_numpy())
    a_late = agreement(late.auto.to_numpy(), late.hand.to_numpy())

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14)
    a = on_weeks(auto)
    ax.fill_between(
        a.index,
        a.worm_index - a["sem"],
        a.worm_index + a["sem"],
        color=ORANGE,
        alpha=0.25,
        lw=0,
    )
    ax.plot(
        a.index,
        a.worm_index,
        color=ORANGE,
        lw=LW,
        label="Automated index (Mon–Fri, ± 1 SEM)",
    )
    h = hand
    ax.fill_between(
        h.index,
        h["mean"] - h["sem"],
        h["mean"] + h["sem"],
        color=BLUE,
        alpha=0.25,
        lw=0,
    )
    ax.plot(
        h.index,
        h["mean"],
        color=BLUE,
        lw=LW,
        marker="o",
        ms=5,
        label="Hand count, Mondays (± 1 SEM)",
    )
    ax.set_ylabel("Worms (per frame)", fontsize=AXIS_PT, fontweight="bold")
    ax.set_xlabel("Date (weekly, UTC)", fontsize=AXIS_PT, fontweight="bold")
    ax.set_ylim(0, 50)
    pad = (END - START) * 0.03
    ax.set_xlim(START - pad, END + pad)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    ax.tick_params(labelsize=TICK_PT, width=1.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.legend(loc="upper left", fontsize=TICK_PT, framealpha=0.9)
    fig.suptitle(
        "Automated vs hand-counted scale worms, Jan 2021–Aug 2023",
        fontsize=TITLE_PT,
        fontweight="bold",
        y=0.96,
    )
    fig.savefig(f"{OUT}.png", dpi=300)
    plt.close(fig)

    caption = (
        "*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are "
        "derived from version-controlled scripts and data.*\n\n"
        "Scale worms per frame at Mushroom vent (ASHES, Axial Seamount), clear OOI CAMHD "
        "footage (CAMHDA301), Jan 2021–10 Aug 2023. Orange: automated index — CSRNet "
        "density head (csrnet_run/best.pt, ÷0.96) read on one frame per recording at the "
        "camera-schedule era's Scene-1 time (420 s to Aug 2021, 540 s to 24 Apr 2022, 210 s "
        "after; 210/420 s in Jan–May 2023), recordings reading < 8.5 dropped as off-scene, "
        "day means averaged to weekly means ± 1 SEM over days, Mon–Fri. Blue: weekly hand "
        "counts by M. Joseph under the written counting rule (Monday Scene-1 frames, mean ± "
        "1 SEM over frames). Gaps = weeks with no data. Agreement on weeks with both "
        f"(automated/hand ratio, Pearson r; bootstrap over weeks, 10 000 resamples, seed 0, "
        f"percentile 95% CIs): all {fmt(all_)}; Jan–Aug 2021 {fmt(a_early)}; Sep 2021–Aug "
        f"2023 {fmt(a_late)}. The index follows the hand-counted timing (2021 spring rise, "
        "late-2022 peak, 2023 decline) but reads early 2021 lower and sparse periods higher "
        "than the hand counts; the off-scene cutoff removes genuinely sparse frames, which "
        "biases low-abundance weeks upward, and Jan–May 2023, where the higher of two "
        "candidate times is kept, reads high (max bias).\n"
    )
    OUT.with_suffix(".caption.md").write_text(caption)
    print(f"wrote {OUT.name}.png + caption")
    print("all", fmt(all_), "\nearly", fmt(a_early), "\nlate", fmt(a_late))


if __name__ == "__main__":
    main()
