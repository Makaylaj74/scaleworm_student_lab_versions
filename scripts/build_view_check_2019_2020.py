"""View check: are MJ's Sep-2026 2019-2020 Scene-1 accepts the same view as 2021-2023 Scene-1?

The 2019-2020 sort (scene_sorting/2019_2020, 191/351 accepted) conflicts with the
2026-09-29 blind reconciliation of Rater B's 2017-2020 sort (2019-2020 judged to lack the
canonical Mushroom-tower Scene-1). Before any 2019-2020 frame is counted, this sheet puts
12 seeded accepts (6 per year, spread across months) under two canonical 2021 Scene-1
references. Candidates are shuffled and lettered so the year is not shown on the sheet;
the key and a blank `same_view` column are in view_check.csv.

    .venv/bin/python3 scripts/build_view_check_2019_2020.py
"""

from __future__ import annotations

import csv
import random
import string
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SORT = REPO / "scene_sorting/2019_2020/sort_log.csv"
SAN = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
OUT = REPO / "validation/view_check_2019_2020"
SEED = 20261111
PER_YEAR = 6
# canonical references: rule-counted 2021 Scene-1 frames (one per 2021 schedule era)
REFS = {
    "CAMHDA301-20210322T211500": REPO / "validation/recount_janaug_2021/images",
    "CAMHDA301-20211108T031500": REPO / "validation/recount_sepdec_2021_rest/images",
}


def pick(rows: list[dict], seed: int = SEED) -> list[dict]:
    """PER_YEAR accepts per year, at most one per month first, chronological within year."""
    rng = random.Random(seed)
    picks = []
    for year in ("2019", "2020"):
        by_month: dict[str, list[dict]] = {}
        for r in rows:
            if r["decision"] == "scene1" and r["stem"][10:14] == year:
                by_month.setdefault(r["stem"][10:16], []).append(r)
        months = sorted(by_month)
        chosen = [
            rng.choice(by_month[m])
            for m in rng.sample(months, min(PER_YEAR, len(months)))
        ]
        pool = [r for m in months for r in by_month[m] if r not in chosen]
        chosen += rng.sample(pool, PER_YEAR - len(chosen))
        picks += sorted(chosen, key=lambda r: r["stem"])
    return picks


def grab(stem: str, t_s: float):
    day = stem[10:18]
    video = SAN / day[:4] / day[4:6] / day[6:] / f"{stem}.mp4"
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read {video} at {t_s}s")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> None:
    with SORT.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    picks = pick(rows)
    rng = random.Random(SEED + 1)
    rng.shuffle(picks)
    letters = string.ascii_uppercase[: len(picks)]

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(24, 14.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (stem, folder) in zip(axes[0, :2], REFS.items(), strict=True):
        ax.imshow(plt.imread(folder / f"{stem}.png"))
        ax.set_title(
            f"REFERENCE 2021 Scene-1 ({stem[10:14]}-{stem[14:16]})",
            fontsize=16,
            fontweight="bold",
            color="#009E73",
        )
    axes[0, 2].text(
        0.02,
        0.5,
        "Is each lettered frame the SAME view\nas the references "
        "(front-on Mushroom tower)?\nRecord y / n / unsure in\n"
        "validation/view_check_2019_2020/view_check.csv",
        fontsize=16,
        va="center",
        transform=axes[0, 2].transAxes,
    )
    for ax, letter, r in zip(axes[1:].flat, letters, picks, strict=True):
        ax.imshow(grab(r["stem"], float(r["scene1_time_s"])))
        ax.set_title(letter, fontsize=20, fontweight="bold")
    fig.subplots_adjust(
        left=0.005, right=0.995, top=0.965, bottom=0.005, wspace=0.02, hspace=0.12
    )
    fig.savefig(OUT / "view_check_2019_2020.png", dpi=150)
    plt.close(fig)

    with (OUT / "view_check.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["letter", "frame_id", "scene1_time_s", "same_view", "notes"])
        for letter, r in zip(letters, picks, strict=True):
            w.writerow([letter, r["stem"], r["scene1_time_s"], "", ""])
    print(
        f"wrote {OUT / 'view_check_2019_2020.png'} and view_check.csv ({len(picks)} frames)"
    )


if __name__ == "__main__":
    main()
