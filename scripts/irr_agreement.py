"""Inter-rater reliability (IRR) statistics for Scene 1 sorting.

Reads each rater's ``sort_log.csv`` (the append-only log written by
:mod:`scene_sorter`; the *last* row for a stem is its final decision) and reports
agreement on two things:

* **Scene 1 vs not** (nominal): Fleiss' kappa across all raters plus pairwise
  Cohen's kappa and raw observed agreement.
* **Scene 1 time** (locator): on recordings *all* raters called Scene 1 with a
  numeric time, the fraction where all raters agree within a tolerance (default
  one 30 s tile) and the mean absolute pairwise time difference.

Every statistic is reported with a bootstrap 95% confidence interval. Bootstrap
convention (stated for reproducibility): resample the set of jointly-labeled
recordings with replacement, ``n_boot`` resamples (default 2000), seeded
(``--seed``, default 20260727), percentile method (2.5 / 97.5).

Skips and recordings a rater has not yet decided are excluded pairwise; the
counts of excluded recordings are reported so coverage is explicit.

Example
-------
::

    python scripts/irr_agreement.py \
        scene_sorting/irr_round_1/raters/rater_a \
        scene_sorting/irr_round_1/raters/rater_b \
        scene_sorting/irr_round_1/raters/rater_c
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np

#: Nominal categories for the binary Scene-1 judgment.
CATEGORIES: tuple[str, ...] = ("scene1", "not_scene1")
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20260727
#: Default time-agreement tolerance: one contact-sheet tile (30 s).
DEFAULT_TOLERANCE_S = 30.0


def load_decisions(sort_log: Path) -> dict[str, dict[str, object]]:
    """Return ``{stem: {"decision", "time"}}`` using the last row per stem.

    ``time`` is a float when the log recorded a numeric ``scene1_time_s`` and
    None otherwise (blank cell, or a non-``scene1`` decision).
    """
    sort_log = Path(sort_log)
    out: dict[str, dict[str, object]] = {}
    if not sort_log.is_file():
        return out
    with sort_log.open(newline="") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("scene1_time_s") or "").strip()
            time: float | None = float(raw) if raw else None
            out[row["stem"]] = {"decision": row["decision"], "time": time}
    return out


def resolve_log(path: Path) -> Path:
    """Accept either a session dir or a direct sort_log.csv path."""
    path = Path(path)
    return path / "sort_log.csv" if path.is_dir() else path


# --- agreement coefficients (implemented locally so bootstrap can recompute) ---


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa for two raters over paired nominal labels."""
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must be the same length")
    n = len(labels_a)
    if n == 0:
        return float("nan")
    cats = CATEGORIES
    po = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / n
    pe = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in cats)
    return float("nan") if pe == 1.0 else (po - pe) / (1.0 - pe)


def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa from an ``N_items x N_categories`` count table.

    Every row must sum to the same number of ratings ``n`` (>= 2).
    """
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[0] == 0:
        return float("nan")
    n_ratings = table.sum(axis=1)
    n = n_ratings[0]
    if n < 2 or not np.allclose(n_ratings, n):
        return float("nan")
    big_n = table.shape[0]
    p_i = (np.square(table).sum(axis=1) - n) / (n * (n - 1))
    p_bar = p_i.mean()
    p_j = table.sum(axis=0) / (big_n * n)
    p_e = np.square(p_j).sum()
    return float("nan") if p_e == 1.0 else float((p_bar - p_e) / (1.0 - p_e))


def _fleiss_table(label_rows: list[list[str]]) -> np.ndarray:
    """Build an ``N x len(CATEGORIES)`` count table from per-item label lists."""
    table = np.zeros((len(label_rows), len(CATEGORIES)), dtype=float)
    for i, labels in enumerate(label_rows):
        for label in labels:
            table[i, CATEGORIES.index(label)] += 1
    return table


def bootstrap_ci(
    items: list,
    statistic,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    percentiles: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, float]:
    """Percentile bootstrap CI: resample ``items`` with replacement ``n_boot`` times."""
    if len(items) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [items[j] for j in rng.choice(idx, size=len(idx), replace=True)]
        value = statistic(sample)
        if not np.isnan(value):
            stats.append(value)
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, percentiles)
    return (float(lo), float(hi))


def compute_report(
    rater_logs: dict[str, dict[str, dict[str, object]]],
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Compute the full IRR report from per-rater decision dicts."""
    raters = list(rater_logs)

    # --- binary Scene1/not: stems every rater gave a terminal (non-skip) call ---
    def terminal(stem: str, log: dict[str, dict[str, object]]) -> str | None:
        rec = log.get(stem)
        return rec["decision"] if rec and rec["decision"] in CATEGORIES else None

    all_stems = sorted(set().union(*(set(log) for log in rater_logs.values())))
    binary_stems = [
        s for s in all_stems if all(terminal(s, rater_logs[r]) for r in raters)
    ]
    label_rows = [[terminal(s, rater_logs[r]) for r in raters] for s in binary_stems]

    observed = (
        sum(len(set(row)) == 1 for row in label_rows) / len(label_rows)
        if label_rows
        else float("nan")
    )
    fleiss = fleiss_kappa(_fleiss_table(label_rows))
    fleiss_ci = bootstrap_ci(
        label_rows, lambda rows: fleiss_kappa(_fleiss_table(rows)), n_boot, seed
    )

    pairwise = {}
    for a, b in itertools.combinations(raters, 2):
        la = [terminal(s, rater_logs[a]) for s in binary_stems]
        lb = [terminal(s, rater_logs[b]) for s in binary_stems]
        pairs = list(zip(la, lb, strict=True))
        ck = cohen_kappa(la, lb)
        ck_ci = bootstrap_ci(
            pairs,
            lambda p: cohen_kappa([x for x, _ in p], [y for _, y in p]),
            n_boot,
            seed,
        )
        pairwise[f"{a}|{b}"] = {"cohen_kappa": ck, "ci95": ck_ci}

    # --- time agreement: stems all raters called scene1 with a numeric time ---
    def scene1_time(stem: str, log: dict[str, dict[str, object]]) -> float | None:
        rec = log.get(stem)
        if rec and rec["decision"] == "scene1" and rec["time"] is not None:
            return float(rec["time"])
        return None

    time_stems = [
        s
        for s in all_stems
        if all(scene1_time(s, rater_logs[r]) is not None for r in raters)
    ]
    time_rows = [[scene1_time(s, rater_logs[r]) for r in raters] for s in time_stems]

    def within_tol_rate(rows: list[list[float]]) -> float:
        if not rows:
            return float("nan")
        return sum(max(r) - min(r) <= tolerance_s for r in rows) / len(rows)

    def mean_abs_diff(rows: list[list[float]]) -> float:
        diffs = [abs(x - y) for r in rows for x, y in itertools.combinations(r, 2)]
        return float(np.mean(diffs)) if diffs else float("nan")

    return {
        "raters": raters,
        "n_boot": n_boot,
        "seed": seed,
        "tolerance_s": tolerance_s,
        "binary": {
            "n_included": len(binary_stems),
            "n_total_stems": len(all_stems),
            "observed_agreement": observed,
            "fleiss_kappa": fleiss,
            "fleiss_kappa_ci95": fleiss_ci,
            "pairwise_cohen": pairwise,
        },
        "time": {
            "n_included": len(time_stems),
            "within_tolerance_rate": within_tol_rate(time_rows),
            "within_tolerance_ci95": bootstrap_ci(
                time_rows, within_tol_rate, n_boot, seed
            ),
            "mean_abs_pairwise_diff_s": mean_abs_diff(time_rows),
            "mean_abs_pairwise_diff_ci95": bootstrap_ci(
                time_rows, mean_abs_diff, n_boot, seed
            ),
        },
    }


def _fmt(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.3f}"


def format_report(report: dict) -> str:
    b, t = report["binary"], report["time"]
    lines = [
        "=" * 64,
        f"Inter-rater reliability — raters: {', '.join(report['raters'])}",
        f"bootstrap: n={report['n_boot']}, seed={report['seed']}, percentile 2.5/97.5",
        "=" * 64,
        "",
        "SCENE 1 vs NOT (nominal)",
        f"  recordings scored by all raters : {b['n_included']} / {b['n_total_stems']}",
        f"  observed agreement (all match)  : {_fmt(b['observed_agreement'])}",
        (
            f"  Fleiss' kappa                   : {_fmt(b['fleiss_kappa'])} "
            f"[{_fmt(b['fleiss_kappa_ci95'][0])}, {_fmt(b['fleiss_kappa_ci95'][1])}]"
        ),
        "  pairwise Cohen's kappa:",
    ]
    for pair, d in b["pairwise_cohen"].items():
        lines.append(
            f"    {pair:24s}: {_fmt(d['cohen_kappa'])} "
            f"[{_fmt(d['ci95'][0])}, {_fmt(d['ci95'][1])}]"
        )
    lines += [
        "",
        f"SCENE 1 TIME (locator, tolerance +-{report['tolerance_s']:g} s)",
        f"  recordings all-scene1 with time : {t['n_included']}",
        (
            f"  within-tolerance agreement rate : {_fmt(t['within_tolerance_rate'])} "
            f"[{_fmt(t['within_tolerance_ci95'][0])}, "
            f"{_fmt(t['within_tolerance_ci95'][1])}]"
        ),
        (
            f"  mean abs pairwise diff (s)      : {_fmt(t['mean_abs_pairwise_diff_s'])} "
            f"[{_fmt(t['mean_abs_pairwise_diff_ci95'][0])}, "
            f"{_fmt(t['mean_abs_pairwise_diff_ci95'][1])}]"
        ),
        "=" * 64,
    ]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "sessions",
        nargs="+",
        type=Path,
        help="Two or more rater session dirs (or sort_log.csv paths).",
    )
    p.add_argument("--tolerance-s", type=float, default=DEFAULT_TOLERANCE_S)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if len(args.sessions) < 2:
        raise SystemExit("need at least two rater sessions to compare")
    rater_logs = {
        resolve_log(s).parent.name or f"rater_{i}": load_decisions(resolve_log(s))
        for i, s in enumerate(args.sessions)
    }
    report = compute_report(
        rater_logs,
        tolerance_s=args.tolerance_s,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
