"""Pull the ASHES Mushroom diffuse-flow thermistor array to a weekly time series.

Source: OOI RS03ASHS-MJ03B-07-TMPSFA301 (24-thermistor diffuse-flow array,
~10-17 m from the Mushroom vent watched by CAMHDA301), on-disk NetCDF under
/home/jovyan/ooi/kdata/...tmpsf_sample/.

For each thermistor (temperature01..24) we apply the OOI QARTOD flag
(keep results in KEEP_FLAGS), resample to weekly means, then summarise the
array. Output: one row per ISO week with per-thermistor weekly means plus
array_mean / array_max / array_p90 and the sample count.

Weekly cadence chosen to match the scaleworm weekly-Monday abundance series.
QARTOD interpretation: 1=pass, 2=not_evaluated, 3=suspect, 4=fail, 9=missing.
We keep {1,2} (pass + not-evaluated) and drop suspect/fail/missing.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

TMPSF_DIR = Path(
    "/home/jovyan/ooi/kdata/RS03ASHS-MJ03B-07-TMPSFA301-streamed-tmpsf_sample"
)
START = pd.Timestamp("2017-01-01", tz="UTC")
END = pd.Timestamp("2025-01-01", tz="UTC")
KEEP_FLAGS = {1, 2}
N_THERM = 24
OUT_CSV = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / ("ashes_thermistor_weekly_2017_2024.csv")
)

_FNAME_RE = re.compile(r"_(\d{8}T\d{6})")


def _file_start(path: str) -> pd.Timestamp | None:
    """Parse the leading YYYYMMDDThhmmss timestamp from a data filename."""
    m = _FNAME_RE.search(Path(path).name)
    if not m:
        return None
    return pd.Timestamp(m.group(1), tz="UTC")


def files_in_range(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Data files whose leading timestamp falls in [start-70d, end).

    The -70d back-pad catches a file that starts before `start` but spills
    into the window (deployment files span up to ~2 months).
    """
    files = sorted(glob.glob(str(TMPSF_DIR / "*tmpsf_sample_*.nc")))
    out = []
    for f in files:
        ts = _file_start(f)
        if ts is None:
            continue
        if (start - pd.Timedelta(days=70)) <= ts < end:
            out.append(f)
    return out


def weekly_from_file(path: str) -> pd.DataFrame | None:
    """Return QC'd weekly-mean temps (one col per thermistor) for one file."""
    cols = [f"temperature{i:02d}" for i in range(1, N_THERM + 1)]
    flagcols = [f"{c}_qartod_results" for c in cols]
    keep = cols + flagcols
    ds = xr.open_dataset(path, drop_variables=None)
    have = [v for v in keep if v in ds.data_vars]
    ds = ds[have]
    df = ds.to_dataframe()
    ds.close()
    df = df.reset_index()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[(df["time"] >= START) & (df["time"] < END)]
    if df.empty:
        return None
    # QC-mask each thermistor against its QARTOD result
    for c in cols:
        fc = f"{c}_qartod_results"
        if c not in df:
            continue
        if fc in df:
            bad = ~df[fc].isin(KEEP_FLAGS)
            df.loc[bad, c] = np.nan
    present = [c for c in cols if c in df]
    wk = (
        df.set_index("time")[present]
        .resample("W-MON", label="left", closed="left")
        .mean()
    )
    wk["_n"] = (
        df.set_index("time")[present[0]]
        .resample("W-MON", label="left", closed="left")
        .count()
    )
    return wk


def main() -> None:
    files = files_in_range(START, END)
    print(f"{len(files)} thermistor files intersect 2021-2024")
    parts = []
    for f in files:
        wk = weekly_from_file(f)
        if wk is not None:
            parts.append(wk)
        print(f"  {Path(f).name[:55]:55s} {'ok' if wk is not None else 'empty'}")
    if not parts:
        raise SystemExit("no data in range")
    # files can overlap week boundaries -> group by week and average
    allwk = pd.concat(parts)
    cols = [f"temperature{i:02d}" for i in range(1, N_THERM + 1)]
    present = [c for c in cols if c in allwk]
    agg = allwk.groupby(level=0).agg({**{c: "mean" for c in present}, "_n": "sum"})
    agg = agg.sort_index()
    tmat = agg[present]
    agg["array_mean"] = tmat.mean(axis=1)
    agg["array_max"] = tmat.max(axis=1)
    agg["array_p90"] = tmat.quantile(0.90, axis=1)
    agg.index.name = "week_start"
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_CSV)
    print(f"\nwrote {OUT_CSV}")
    print(f"weeks: {len(agg)}  {agg.index.min().date()} .. {agg.index.max().date()}")
    print(agg[["array_mean", "array_max", "array_p90", "_n"]].describe())


if __name__ == "__main__":
    main()
