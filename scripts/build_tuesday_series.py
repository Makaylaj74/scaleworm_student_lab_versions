"""Build the Tuesday scale-worm abundance series (2021-2023).

Two series over the same 2021-09 -> 2023-08 window, disjoint in time:

1. **Manual** (primary, ground truth): the 49 Tuesdays MJ hand-counted every
   Scene-1 slot. Per Tuesday = mean +/- SEM (and a Student-t 95% CI) across its
   up-to-8 three-hourly slots. No model, no correction.

2. **Density-corrected AI** (secondary, caveated): the 43 *other* Tuesdays that
   the v2 detector scored but nobody hand-counted. v2 saturates on dense scenes
   (it recovers only ~38% of worms and its per-frame count explains just ~20% of
   the manual variance), so a scalar recall factor is invalid. Instead we fit an
   ordinary-least-squares calibration ``manual ~ ai`` on the 229 paired frames
   and apply it per slot, propagating BOTH the fit uncertainty and the residual
   scatter through a paired bootstrap into a per-Tuesday 95% CI. The wide bars
   are the honest cost of using the detector where no hand count exists.

The two Tuesday sets do not overlap, so the extended series is their union:
tight manual anchors interleaved with wide corrected-AI fill-ins.

Inputs
  validation/tuesday_manual_series/box_correct_manifest.csv  (MJ box counts)
  notebooks/ai_counts_tuesday_2021_2023.csv                  (raw v2 boxes @ conf 0.25)

Outputs
  validation/tuesday_manual_series/tuesday_manual_timeseries.csv
  notebooks/tuesday_corrected_ai_timeseries.csv
  notebooks/tuesday_extended_timeseries.csv
  notebooks/tuesday_ai_vs_manual_pairs.csv       (229 paired frames, for the scatter)
  notebooks/tuesday_calibration.json             (fit coefficients + diagnostics)

Uncertainty conventions (stated per lab "Defensible Statistics" rule), both
percentile bootstrap 95% CIs (n=10000, seed 20260922) so the two series are
comparable on one axis and both respect the count support [0, inf):
  - Manual per-Tuesday: bootstrap over that day's counted slots; blank for n=1.
  - Corrected per-Tuesday: paired bootstrap that resamples the 229 pairs, refits
    OLS, and adds a resampled residual per predicted slot (counts clipped at 0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MANUAL_MANIFEST = REPO / "validation/tuesday_manual_series/box_correct_manifest.csv"
AI_COUNTS = REPO / "notebooks/ai_counts_tuesday_2021_2023.csv"

OUT_MANUAL = REPO / "validation/tuesday_manual_series/tuesday_manual_timeseries.csv"
OUT_CORRECTED = REPO / "notebooks/tuesday_corrected_ai_timeseries.csv"
OUT_EXTENDED = REPO / "notebooks/tuesday_extended_timeseries.csv"
OUT_PAIRS = REPO / "notebooks/tuesday_ai_vs_manual_pairs.csv"
OUT_CALIB = REPO / "notebooks/tuesday_calibration.json"

N_BOOT = 10_000
SEED = 20260922


class OLSFit:
    """Simple univariate OLS ``y = b0 + b1 * x`` with fit diagnostics."""

    def __init__(self, b0: float, b1: float, resid: np.ndarray, r2: float, rse: float):
        self.b0 = b0
        self.b1 = b1
        self.resid = resid
        self.r2 = r2
        self.rse = rse

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.b0 + self.b1 * np.asarray(x, dtype=float)


def fit_ols(x: np.ndarray, y: np.ndarray) -> OLSFit:
    """Fit ``y ~ x`` by ordinary least squares and return coefficients + diagnostics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 points to fit with residual diagnostics")
    xbar, ybar = x.mean(), y.mean()
    sxx = np.sum((x - xbar) ** 2)
    sxy = np.sum((x - xbar) * (y - ybar))
    b1 = sxy / sxx
    b0 = ybar - b1 * xbar
    resid = y - (b0 + b1 * x)
    sse = np.sum(resid**2)
    sst = np.sum((y - ybar) ** 2)
    r2 = 1.0 - sse / sst
    rse = float(np.sqrt(sse / (n - 2)))
    return OLSFit(float(b0), float(b1), resid, float(r2), rse)


def load_pairs(manual_manifest: Path, ai_counts: Path) -> pd.DataFrame:
    """Return the frames that have BOTH a manual (counted) count and an AI count."""
    man = pd.read_csv(manual_manifest)
    man = man[(man["frame_status"] == "counted") & man["worm_count"].notna()].copy()
    man["worm_count"] = man["worm_count"].astype(int)
    ai = pd.read_csv(ai_counts)
    merged = man.merge(ai[["frame_id", "ai_count"]], on="frame_id", how="inner")
    merged["date"] = merged["datetime_utc"].str.slice(0, 10)
    return merged[["frame_id", "date", "ai_count", "worm_count"]].reset_index(drop=True)


def load_ai_only(manual_manifest: Path, ai_counts: Path) -> pd.DataFrame:
    """Return AI-scored frames that were never hand-labeled (counted or blur)."""
    man = pd.read_csv(manual_manifest)
    labeled = set(
        man[man["frame_status"].isin(["counted", "unusable_blur"])]["frame_id"]
    )
    ai = pd.read_csv(ai_counts)
    extra = ai[~ai["frame_id"].isin(labeled)].copy()
    return extra[["frame_id", "date", "ai_count"]].reset_index(drop=True)


def aggregate_manual(
    manual_manifest: Path, n_boot: int = N_BOOT, seed: int = SEED
) -> pd.DataFrame:
    """Per-Tuesday mean/SEM and a bootstrap 95% CI of the hand counts.

    The daily-mean CI is a percentile bootstrap over that day's slots (same method
    as the corrected series, so the two are comparable on one axis, and the count
    support [0, inf) is respected -- a Student-t interval goes negative and explodes
    for the single 2-slot Tuesday). CI is blank for single-slot days (n=1).
    """
    rng = np.random.default_rng(seed)
    man = pd.read_csv(manual_manifest)
    man = man[(man["frame_status"] == "counted") & man["worm_count"].notna()].copy()
    man["worm_count"] = man["worm_count"].astype(int)
    man["date"] = man["datetime_utc"].str.slice(0, 10)
    rows = []
    for date, grp in man.groupby("date"):
        c = grp["worm_count"].to_numpy()
        n = c.size
        mean = float(c.mean())
        if n > 1:
            sem = float(c.std(ddof=1) / np.sqrt(n))
            boot = rng.choice(c, size=(n_boot, n), replace=True).mean(axis=1)
            lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
        else:
            sem, lo, hi = np.nan, np.nan, np.nan
        rows.append(
            {
                "date": date,
                "n_slots": int(n),
                "mean_worms": round(mean, 3),
                "sem_worms": round(sem, 3) if not np.isnan(sem) else "",
                "ci_lo": round(lo, 3) if not np.isnan(lo) else "",
                "ci_hi": round(hi, 3) if not np.isnan(hi) else "",
                "total_worms": int(c.sum()),
                "source": "manual",
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def bootstrap_corrected(
    pairs: pd.DataFrame,
    ai_only: pd.DataFrame,
    fit: OLSFit,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Per-Tuesday corrected-AI mean with a paired-bootstrap percentile 95% CI.

    Point estimate uses the full-data fit; the CI resamples the paired frames,
    refits OLS, and adds a resampled residual per predicted slot (clipped at 0).
    """
    rng = np.random.default_rng(seed)
    x = pairs["ai_count"].to_numpy(dtype=float)
    y = pairs["worm_count"].to_numpy(dtype=float)
    n = x.size

    dates = sorted(ai_only["date"].unique())
    slots = {
        d: ai_only.loc[ai_only["date"] == d, "ai_count"].to_numpy(float) for d in dates
    }

    # Point estimate: full-data prediction, clipped at 0, averaged per date.
    point = {d: float(np.clip(fit.predict(slots[d]), 0, None).mean()) for d in dates}

    boot = {d: np.empty(n_boot) for d in dates}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        xbar = xb.mean()
        sxx = np.sum((xb - xbar) ** 2)
        if sxx == 0:
            # Degenerate resample (all AI values identical): fall back to a
            # flat fit at the resample mean. Effectively impossible at n=229;
            # guards the small-sample path only.
            b1, b0 = 0.0, yb.mean()
        else:
            b1 = np.sum((xb - xbar) * (yb - yb.mean())) / sxx
            b0 = yb.mean() - b1 * xbar
        resid_b = yb - (b0 + b1 * xb)
        for d in dates:
            a = slots[d]
            e = rng.choice(resid_b, size=a.size, replace=True)
            boot[d][b] = np.clip(b0 + b1 * a + e, 0, None).mean()

    rows = []
    for d in dates:
        lo, hi = np.percentile(boot[d], [2.5, 97.5])
        rows.append(
            {
                "date": d,
                "n_slots": int(slots[d].size),
                "mean_worms": round(point[d], 3),
                "sem_worms": "",
                "ci_lo": round(float(lo), 3),
                "ci_hi": round(float(hi), 3),
                "total_worms": "",
                "source": "ai_corrected",
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manual-manifest", type=Path, default=MANUAL_MANIFEST)
    ap.add_argument("--ai-counts", type=Path, default=AI_COUNTS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    pairs = load_pairs(args.manual_manifest, args.ai_counts)
    ai_only = load_ai_only(args.manual_manifest, args.ai_counts)
    fit = fit_ols(pairs["ai_count"].to_numpy(), pairs["worm_count"].to_numpy())

    manual = aggregate_manual(args.manual_manifest, args.n_boot, args.seed)
    corrected = bootstrap_corrected(pairs, ai_only, fit, args.n_boot, args.seed)
    extended = pd.concat([manual, corrected], ignore_index=True).sort_values("date")

    agg_recall = pairs["ai_count"].sum() / pairs["worm_count"].sum()
    calib = {
        "model": "OLS manual ~ ai",
        "b0": round(fit.b0, 4),
        "b1": round(fit.b1, 4),
        "r2": round(fit.r2, 4),
        "residual_sd": round(fit.rse, 4),
        "n_pairs": len(pairs),
        "aggregate_recall_ai_over_manual": round(float(agg_recall), 4),
        "ai_support_min": int(pairs["ai_count"].min()),
        "ai_support_max": int(pairs["ai_count"].max()),
        "n_ai_only_frames": len(ai_only),
        "n_manual_tuesdays": len(manual),
        "n_corrected_tuesdays": len(corrected),
        "bootstrap_n": int(args.n_boot),
        "bootstrap_seed": int(args.seed),
        "bootstrap_method": "paired resample + residual resample, percentile 95% CI",
    }

    for path, df in [
        (OUT_MANUAL, manual),
        (OUT_CORRECTED, corrected),
        (OUT_EXTENDED, extended),
        (OUT_PAIRS, pairs),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    OUT_CALIB.write_text(json.dumps(calib, indent=2))

    print(
        f"calibration: manual = {fit.b0:.2f} + {fit.b1:.3f}*ai  "
        f"R2={fit.r2:.3f}  residual_SD={fit.rse:.1f}  n={len(pairs)}\n"
        f"aggregate v2 recall = {agg_recall:.1%}\n"
        f"manual Tuesdays: {len(manual)}  corrected-AI Tuesdays: {len(corrected)}  "
        f"extended: {len(extended)}\n"
        f"wrote {OUT_MANUAL.name}, {OUT_CORRECTED.name}, {OUT_EXTENDED.name}, "
        f"{OUT_PAIRS.name}, {OUT_CALIB.name}"
    )


if __name__ == "__main__":
    main()
