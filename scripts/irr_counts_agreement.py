"""Inter-rater reliability for the manual scaleworm *count/box* annotations.

Round-1 IRR (`scripts/irr_agreement.py`) scored a **categorical** scene-sorting
decision (Scene-1 vs not) with Fleiss'/Cohen's kappa. This is a different task: two
or more raters independently box every worm in the same frames, so the quantity is a
**count** (and a set of boxes), not a category. Kappa does not apply; this script
reports the agreement statistics that do.

Each rater returns a directory shaped exactly like the packet they were given::

    raters/<rater_id>/
        irr_manifest.csv        # schema rows; frame_status, worm_count, counter
        labels/<frame_id>.txt   # YOLO boxes, one line per worm (class cx cy w h)

`worm_count` is the number of boxes (schema rule 1), so the label files are the
source of truth for counts; the manifest supplies `frame_status` (a frame only
enters the analysis when EVERY rater marked it `counted` -- an `unusable_blur` /
`no_scene1` frame is missing data, not a zero, and is excluded pairwise).

Two families of agreement, each with a bootstrap 95% CI (percentile, resampling over
the jointly-counted frames):

* **Count level** -- do the raters get the same number of worms?
  ICC(2,1) absolute agreement across all raters; per pair: mean absolute count
  difference, bias, Pearson r, and exact-count agreement fraction.
* **Box level** -- do they mark the *same* worms, not just the same total?
  Per pair, a greedy centre-in-box match (in normalised coordinates -- the click
  boxes are a fixed median size, so IoU would be unfair, matching Round-1's
  `eval_val_recall.py`) yields precision/recall/F1 and the matched fraction.

Usage::

    python scripts/irr_counts_agreement.py \
        validation/irr_round_2/raters/MJ \
        validation/irr_round_2/raters/LG \
        validation/irr_round_2/raters/MS
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

SEED = 20260925
N_BOOT = 10000
COUNTED = "counted"


def load_rater(rater_dir: Path) -> dict[str, dict]:
    """Return {frame_id: {"status": str, "boxes": [(cx,cy,w,h), ...]}} for one rater.

    `status` comes from `irr_manifest.csv`; `boxes` are the normalised YOLO boxes in
    `labels/<frame_id>.txt`. A frame present in the manifest but missing a label file
    is treated as zero boxes (only meaningful when status == "counted").
    """
    rater_dir = Path(rater_dir)
    manifest = rater_dir / "irr_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"no irr_manifest.csv in {rater_dir}")
    labels_dir = rater_dir / "labels"
    out: dict[str, dict] = {}
    for row in csv.DictReader(manifest.open(newline="")):
        fid = row["frame_id"]
        out[fid] = {
            "status": (row.get("frame_status") or "").strip(),
            "boxes": load_boxes(labels_dir / f"{fid}.txt"),
        }
    return out


def load_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    """Read a YOLO label file -> list of normalised (cx, cy, w, h) worm boxes."""
    boxes: list[tuple[float, float, float, float]] = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append(tuple(float(v) for v in parts[1:]))  # type: ignore[arg-type]
    return boxes


def _to_xyxy(box: tuple[float, float, float, float]):
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def match_boxes(boxes_a, boxes_b) -> int:
    """Greedy centre-in-box match between two rater box sets; return #matched pairs.

    An A-box matches a B-box when the A-centre lies inside the B-box; each B-box is
    claimed at most once. Symmetric enough for IRR (F1 is orientation-independent).
    """
    b_xyxy = [_to_xyxy(b) for b in boxes_b]
    claimed = [False] * len(b_xyxy)
    matched = 0
    for cx, cy, _w, _h in boxes_a:
        for j, (x1, y1, x2, y2) in enumerate(b_xyxy):
            if not claimed[j] and x1 <= cx <= x2 and y1 <= cy <= y2:
                claimed[j] = True
                matched += 1
                break
    return matched


def icc_a1(matrix: np.ndarray) -> float:
    """ICC(A,1): two-way random effects, single measures, absolute agreement.

    `matrix` is (n_subjects x k_raters). McGraw & Wong (1996) formulation. Returns
    NaN when it is undefined (fewer than 2 subjects/raters or zero total variance).
    """
    x = np.asarray(matrix, dtype=float)
    n, k = x.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    ss_total = ((x - grand) ** 2).sum()
    if ss_total == 0:
        return 1.0  # perfect agreement (all identical)
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if denom == 0:
        return float("nan")
    return (msr - mse) / denom


def _boot_ci(values: np.ndarray, stat, rng, n_boot: int = N_BOOT):
    """Percentile 95% CI of `stat` over rows of `values` via bootstrap resampling."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = np.array([stat(arr[i]) for i in idx])
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def compute(rater_dirs: list[Path], seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """Compute the full IRR report for >=2 rater directories. Returns a metrics dict."""
    if len(rater_dirs) < 2:
        raise ValueError("need at least two rater directories")
    names = [Path(d).name for d in rater_dirs]
    data = {name: load_rater(d) for name, d in zip(names, rater_dirs)}

    # Joint set: frames every rater marked `counted`.
    per_rater_counted = {
        name: {fid for fid, v in d.items() if v["status"] == COUNTED}
        for name, d in data.items()
    }
    joint = sorted(set.intersection(*per_rater_counted.values()))
    rng = np.random.default_rng(seed)

    # counts[name] aligned to `joint`
    counts = {
        name: np.array([len(data[name][fid]["boxes"]) for fid in joint], dtype=float)
        for name in names
    }

    result: dict = {
        "raters": names,
        "n_joint": len(joint),
        "joint_frames": joint,
        "per_rater_counted": {n: len(s) for n, s in per_rater_counted.items()},
        "excluded": {
            n: sorted(per_rater_counted[n] - set(joint)) for n in names
        },
        "count": {},
        "box": {},
    }
    if not joint:
        return result

    # ---- count-level ----
    matrix = np.column_stack([counts[n] for n in names])
    icc = icc_a1(matrix)
    icc_lo, icc_hi = _boot_ci(
        np.arange(len(joint)),
        lambda idx: icc_a1(matrix[idx.astype(int)]),
        rng,
        n_boot,
    )
    all_equal = np.array(
        [len({counts[n][i] for n in names}) == 1 for i in range(len(joint))],
        dtype=float,
    )
    result["count"]["icc_a1"] = icc
    result["count"]["icc_a1_ci"] = (icc_lo, icc_hi)
    result["count"]["exact_all"] = float(all_equal.mean())
    result["count"]["totals"] = {n: int(counts[n].sum()) for n in names}
    result["count"]["mean_per_frame"] = {n: float(counts[n].mean()) for n in names}

    result["count"]["pairwise"] = {}
    for a, b in itertools.combinations(names, 2):
        diff = counts[a] - counts[b]
        absdiff = np.abs(diff)
        mad_ci = _boot_ci(absdiff, np.mean, rng, n_boot)
        bias_ci = _boot_ci(diff, np.mean, rng, n_boot)
        exact = (absdiff == 0).astype(float)
        # Pearson r (guard zero variance)
        if counts[a].std() > 0 and counts[b].std() > 0:
            r = float(np.corrcoef(counts[a], counts[b])[0, 1])
        else:
            r = float("nan")
        result["count"]["pairwise"][f"{a}|{b}"] = {
            "mean_abs_diff": float(absdiff.mean()),
            "mean_abs_diff_ci": mad_ci,
            "bias": float(diff.mean()),
            "bias_ci": bias_ci,
            "exact_agree": float(exact.mean()),
            "pearson_r": r,
        }

    # ---- box-level ----
    result["box"]["pairwise"] = {}
    for a, b in itertools.combinations(names, 2):
        per_frame_f1 = []
        tp_tot = na_tot = nb_tot = 0
        for fid in joint:
            ba = data[a][fid]["boxes"]
            bb = data[b][fid]["boxes"]
            tp = match_boxes(ba, bb)
            tp_tot += tp
            na_tot += len(ba)
            nb_tot += len(bb)
            denom = len(ba) + len(bb)
            per_frame_f1.append(1.0 if denom == 0 else 2 * tp / denom)
        f1_arr = np.array(per_frame_f1, dtype=float)
        micro_p = tp_tot / na_tot if na_tot else float("nan")
        micro_r = tp_tot / nb_tot if nb_tot else float("nan")
        micro_f1 = (
            2 * tp_tot / (na_tot + nb_tot) if (na_tot + nb_tot) else float("nan")
        )
        result["box"]["pairwise"][f"{a}|{b}"] = {
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "mean_frame_f1": float(f1_arr.mean()),
            "mean_frame_f1_ci": _boot_ci(f1_arr, np.mean, rng, n_boot),
        }
    return result


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def print_report(res: dict) -> None:
    names = res["raters"]
    print("=" * 70)
    print("Manual scaleworm count/box IRR")
    print("=" * 70)
    print(f"raters: {', '.join(names)}")
    print("counted per rater: " + ", ".join(
        f"{n}={res['per_rater_counted'][n]}" for n in names))
    print(f"jointly counted (analysis set): {res['n_joint']} frames")
    for n in names:
        ex = res["excluded"][n]
        if ex:
            print(f"  excluded (not joint) for {n}: {len(ex)} -> {', '.join(ex)}")
    if not res["n_joint"]:
        print("\nNo jointly-counted frames -- nothing to compare.")
        return

    c = res["count"]
    print("\n-- count level --")
    print("totals: " + ", ".join(f"{n}={c['totals'][n]}" for n in names))
    print("mean/frame: " + ", ".join(
        f"{n}={c['mean_per_frame'][n]:.2f}" for n in names))
    print(f"ICC(A,1) absolute agreement: {c['icc_a1']:.3f} "
          f"{_fmt_ci(c['icc_a1_ci'])}")
    print(f"exact-count agreement (all raters identical): {c['exact_all']:.1%}")
    for pair, m in c["pairwise"].items():
        print(f"  {pair}: mean|Δ|={m['mean_abs_diff']:.2f} "
              f"{_fmt_ci(m['mean_abs_diff_ci'])}  bias={m['bias']:+.2f} "
              f"{_fmt_ci(m['bias_ci'])}  r={m['pearson_r']:.3f}  "
              f"exact={m['exact_agree']:.0%}")

    print("\n-- box level (do they mark the same worms?) --")
    for pair, m in res["box"]["pairwise"].items():
        print(f"  {pair}: F1={m['micro_f1']:.2f} (P={m['micro_precision']:.2f} "
              f"R={m['micro_recall']:.2f})  mean-frame-F1={m['mean_frame_f1']:.2f} "
              f"{_fmt_ci(m['mean_frame_f1_ci'])}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rater_dirs", nargs="+", type=Path,
                    help="one directory per rater (each with irr_manifest.csv + labels/)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--nboot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    res = compute(args.rater_dirs, seed=args.seed, n_boot=args.nboot)
    print_report(res)


if __name__ == "__main__":
    main()
