"""Draw a stratified inter-rater-reliability (IRR) subset and build blinded rater dirs.

Given the ``manifest.csv`` produced by :mod:`scene_sampler`, this selects a
reproducible random subset of recordings stratified by ``year x slot`` (so both
years and all eight UTC slots are represented), then lays out one *blinded*
sorting session per rater: each rater gets an identical copy of the selected
contact sheets under their own ``contact_sheets/`` folder and logs decisions to
their own ``sort_log.csv``. Raters never see each other's decisions.

The *draw* (which recordings are selected) depends only on the manifest and the
seed, so it is fully reproducible. The *copy* step locates each sheet wherever it
currently sits in the source session (``contact_sheets/``, ``scene1/`` or
``not_scene1/``), so it works even after the source session has been partly
sorted and never disturbs those piles.

Example
-------
::

    python scripts/draw_irr_subset.py \
        --manifest   scene_sorting/full_2023_2024/manifest.csv \
        --source-session scene_sorting/full_2023_2024 \
        --out        scene_sorting/irr_round_1 \
        --n 80 --seed 20260727 \
        --raters rater_a rater_b rater_c
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

#: Default RNG seed for the draw. Stated explicitly so the subset is reproducible.
DEFAULT_SEED = 20260727
#: Sub-folders of a source session that may hold a contact sheet after sorting.
SOURCE_SUBDIRS: tuple[str, ...] = ("contact_sheets", "scene1", "not_scene1")


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load a sampler manifest and add an integer ``year`` column.

    Only rows whose video actually exists (``video_exists`` truthy) are eligible
    for the draw; the returned frame is filtered to those rows.
    """
    # slot is a zero-padded token ("001500") — keep it a string so the leading
    # zeros survive (pandas would otherwise infer int and drop them).
    df = pd.read_csv(manifest_path, dtype={"slot": str})
    df = df[df["video_exists"].astype(str).str.lower() == "true"].copy()
    df["year"] = pd.to_datetime(df["utc_datetime"], utc=True).dt.year
    return df.reset_index(drop=True)


def allocate(n_target: int, n_strata: int) -> list[int]:
    """Split ``n_target`` as evenly as possible across ``n_strata`` (largest first).

    Returns a list of per-stratum quotas whose sum is ``n_target``; the first
    ``n_target % n_strata`` quotas are one larger than the rest.
    """
    if n_strata <= 0:
        raise ValueError("n_strata must be positive")
    base, rem = divmod(n_target, n_strata)
    return [base + 1 if i < rem else base for i in range(n_strata)]


def draw_subset(
    manifest: pd.DataFrame, n_target: int, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """Select a reproducible ``year x slot`` stratified subset of recordings.

    Strata are visited in a fixed (sorted) order and each is allocated an even
    share of ``n_target`` (see :func:`allocate`). Within a stratum, stems are
    sorted and sampled without replacement using a seeded RNG. If a stratum holds
    fewer rows than its quota, all of its rows are taken (the result is then
    slightly under ``n_target``); shortfalls are not redistributed, keeping the
    draw a pure function of ``(manifest, n_target, seed)``.

    Returns the selected rows with a ``stratum`` column, sorted by stem.
    """
    rng = np.random.default_rng(seed)
    strata = sorted(manifest.groupby(["year", "slot"]).groups.keys())
    quotas = allocate(n_target, len(strata))

    picks: list[pd.DataFrame] = []
    for (year, slot), quota in zip(strata, quotas, strict=True):
        rows = manifest[(manifest["year"] == year) & (manifest["slot"] == slot)]
        rows = rows.sort_values("stem")
        take = min(quota, len(rows))
        if take == 0:
            continue
        idx = rng.choice(len(rows), size=take, replace=False)
        chosen = rows.iloc[np.sort(idx)].copy()
        chosen["stratum"] = f"{year}_{slot}"
        picks.append(chosen)

    subset = pd.concat(picks, ignore_index=True) if picks else manifest.iloc[0:0].copy()
    return subset.sort_values("stem").reset_index(drop=True)


def find_sheet(stem: str, source_session: Path) -> Path | None:
    """Locate ``<stem>.png`` in a source session, wherever it has been sorted to."""
    for sub in SOURCE_SUBDIRS:
        candidate = source_session / sub / f"{stem}.png"
        if candidate.is_file():
            return candidate
    return None


def build_rater_dirs(
    subset: pd.DataFrame,
    out_root: Path,
    source_session: Path,
    rater_ids: list[str],
) -> dict[str, object]:
    """Write ``irr_subset.csv`` and populate one blinded session per rater.

    Each rater dir gets ``contact_sheets/`` (copies of every located sheet),
    empty ``scene1/`` and ``not_scene1/`` folders, and no log yet (created on
    first decision). Returns a summary dict: copied stems, missing stems, and the
    per-rater contact-sheet directories.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_root / "irr_subset.csv", index=False)

    located: dict[str, Path] = {}
    missing: list[str] = []
    for stem in subset["stem"]:
        src = find_sheet(stem, Path(source_session))
        if src is None:
            missing.append(stem)
        else:
            located[stem] = src

    rater_sheet_dirs: dict[str, Path] = {}
    for rater in rater_ids:
        rater_root = out_root / "raters" / rater
        sheets = rater_root / "contact_sheets"
        for sub in ("contact_sheets", "scene1", "not_scene1"):
            (rater_root / sub).mkdir(parents=True, exist_ok=True)
        for stem, src in located.items():
            shutil.copy2(src, sheets / f"{stem}.png")
        rater_sheet_dirs[rater] = sheets

    return {
        "copied": sorted(located),
        "missing": missing,
        "rater_sheet_dirs": rater_sheet_dirs,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--source-session",
        type=Path,
        required=True,
        help="Session dir holding the rendered contact sheets (searched across "
        "contact_sheets/, scene1/, not_scene1/).",
    )
    p.add_argument("--out", type=Path, required=True, help="IRR round output dir.")
    p.add_argument("--n", type=int, default=80, help="Target subset size.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--raters", nargs="+", default=["rater_a", "rater_b", "rater_c"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = load_manifest(args.manifest)
    subset = draw_subset(manifest, n_target=args.n, seed=args.seed)
    summary = build_rater_dirs(subset, args.out, args.source_session, args.raters)

    print(f"Manifest eligible recordings : {len(manifest)}")
    print(f"Subset drawn (seed={args.seed}) : {len(subset)}  (target {args.n})")
    print("Per-stratum counts:")
    print(subset["stratum"].value_counts().sort_index().to_string())
    print(f"Sheets copied to each rater  : {len(summary['copied'])}")
    if summary["missing"]:
        print(f"WARNING: {len(summary['missing'])} sheet(s) not found on disk:")
        for stem in summary["missing"]:
            print(f"  missing: {stem}")
    print(f"\nSubset table: {args.out / 'irr_subset.csv'}")
    print("Point notebook 25 SESSION_DIR at each of:")
    for rater, sheets in summary["rater_sheet_dirs"].items():
        print(f"  {rater}: {sheets.parent}")


if __name__ == "__main__":
    main()
