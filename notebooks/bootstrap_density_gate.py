"""Bootstrap confidence intervals for the CSRNet density-head gate.

Defensible Statistics: every ratio/metric in the gate is a point estimate over
n=87 val frames, with some density bins as small as n=6. This attaches
percentile bootstrap CIs so the dense-bin claims are not read as precise.

Convention: non-parametric bootstrap, resampling frames with replacement.
n_boot = 10000, seed = 0, 95% percentile interval (2.5 / 97.5). Per-density-bin
ratios resample within the bin (conditional on the bin), matching how the point
estimates were computed; overall ratio/MAE/r resample all 87 frames.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "csrnet_run" / "gate_density_results.csv"
N_BOOT = 10000
SEED = 0
BINS = [(0, 10), (10, 20), (20, 30), (30, 999)]


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, d, v = [], [], []
    for row in csv.DictReader(RESULTS.open()):
        m.append(float(row["manual"]))
        d.append(float(row["density_count"]))
        v.append(float(row["v5_corr"]))
    return np.array(m), np.array(d), np.array(v)


def ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def boot_ratio(pred: np.ndarray, manual: np.ndarray, rng) -> np.ndarray:
    """Mean per-frame ratio (pred/manual), bootstrapped over frames."""
    n = len(manual)
    out = np.empty(N_BOOT)
    ratios = pred / manual
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        out[i] = ratios[idx].mean()
    return out


def boot_stat(pred: np.ndarray, manual: np.ndarray, rng, fn) -> np.ndarray:
    n = len(manual)
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        out[i] = fn(pred[idx], manual[idx])
    return out


def main() -> None:
    m, d, v = load()

    mae = lambda p, t: np.abs(p - t).mean()
    overall_ratio = lambda p, t: p.sum() / t.sum()
    pearson = lambda p, t: np.corrcoef(p, t)[0, 1]

    print(
        f"bootstrap gate CIs  (val n={len(m)}, n_boot={N_BOOT}, seed={SEED}, "
        "95% percentile)\n"
    )

    print("overall metrics [point (95% CI)]:")
    for name, p in [("density integral", d), ("v5 count/0.72", v)]:
        lo_r, hi_r = ci(boot_stat(p, m, np.random.default_rng(SEED), pearson))
        lo_m, hi_m = ci(boot_stat(p, m, np.random.default_rng(SEED), mae))
        lo_o, hi_o = ci(boot_stat(p, m, np.random.default_rng(SEED), overall_ratio))
        print(
            f"  {name:<18}  r {pearson(p, m):.3f} ({lo_r:.3f},{hi_r:.3f})  "
            f"MAE {mae(p, m):.2f} ({lo_m:.2f},{hi_m:.2f})  "
            f"ratio {overall_ratio(p, m):.2f}x ({lo_o:.2f},{hi_o:.2f})"
        )

    print("\nper-bin mean ratio to manual [point (95% CI)]:")
    for lo, hi in BINS:
        mb = (m >= lo) & (m < hi)
        n = int(mb.sum())
        if n < 1:
            continue
        dr = (d[mb] / m[mb]).mean()
        vr = (v[mb] / m[mb]).mean()
        dlo, dhi = ci(boot_ratio(d[mb], m[mb], np.random.default_rng(SEED)))
        vlo, vhi = ci(boot_ratio(v[mb], m[mb], np.random.default_rng(SEED)))
        print(
            f"  manual {lo:2d}-{hi:<3d} (n={n:2d}):  "
            f"density {dr:.2f}x ({dlo:.2f},{dhi:.2f})   "
            f"v5/0.72 {vr:.2f}x ({vlo:.2f},{vhi:.2f})"
        )


if __name__ == "__main__":
    main()
