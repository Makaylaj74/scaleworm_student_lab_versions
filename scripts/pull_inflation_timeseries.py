"""Weekly Axial caldera magma-inflation series (2021-2024) from BOTPT.

Source: OOI RS03CCAL-MJ03F-05-BOTPTA301 (Central Caldera bottom-pressure/tilt,
the caldera-center inflation station ~2 km from ASHES/Mushroom), the
`botpt_nano_sample_24hour` stream, which carries the OOI BOTSFLU de-tided
seafloor-depth products:
  - botsflu_daydepth (m)   : daily de-tided seafloor depth
  - botsflu_8wkrate (cm/yr): 8-week seafloor uplift rate

Magma recharge inflates the caldera floor -> seafloor depth DECREASES (rises).
We convert to positive-up signals for readability:
  - uplift_cm      : cumulative uplift since the 2021-01 reference
                     = (ref_depth - daydepth) * 100  (positive = inflating)
  - uplift_rate    : botsflu_8wkrate (cm/yr; positive = inflating)
Aggregated to weekly means to match the thermistor/seismicity/worm cadence.
QC: keep botsflu_daydepth where its QARTOD result is in KEEP_FLAGS.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BOTPT_DIR = Path(
    "/home/jovyan/ooi/kdata/"
    "RS03CCAL-MJ03F-05-BOTPTA301-streamed-botpt_nano_sample_24hour"
)
START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2025-01-01")
# BOTSFLU is a derived product: its QARTOD result is 0 (not evaluated) rather
# than 1/2, so keep 0 too; only explicit suspect/fail/missing (3/4/9) drop.
KEEP_FLAGS = {0, 1, 2}
OUT_CSV = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / ("axial_inflation_weekly_2017_2024.csv")
)


def load_daydepth() -> pd.DataFrame:
    """Concatenate all 24h files -> daily de-tided depth + 8wk uplift rate."""
    files = sorted(glob.glob(str(BOTPT_DIR / "*.nc")))
    keep = [
        "botsflu_daydepth",
        "botsflu_daydepth_qc_results",
        "botsflu_8wkrate",
    ]
    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        have = [v for v in keep if v in ds.data_vars]
        df = ds[have].to_dataframe().reset_index()
        ds.close()
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    # QC-mask depth
    if "botsflu_daydepth_qc_results" in df:
        bad = ~df["botsflu_daydepth_qc_results"].isin(KEEP_FLAGS)
        df.loc[bad, "botsflu_daydepth"] = np.nan
    df = (
        df.dropna(subset=["botsflu_daydepth"])
        .drop_duplicates(subset="time")
        .sort_values("time")
        .set_index("time")
    )
    return df


def main() -> None:
    df = load_daydepth()
    print(
        f"BOTPT daily depth: {len(df):,} days "
        f"{df.index.min().date()} .. {df.index.max().date()}"
    )
    sub = df[(df.index >= START) & (df.index < END)]
    # reference depth = mean of the first available 2 weeks in 2021.
    # daydepth is stored as negative elevation (m); a LARGER (less negative)
    # value = shallower seafloor = uplift, so uplift = (daydepth - ref).
    ref = sub["botsflu_daydepth"].iloc[:14].mean()
    weekly = pd.DataFrame(
        {
            "daydepth_m": sub["botsflu_daydepth"]
            .resample("W-MON", label="left", closed="left")
            .mean(),
        }
    )
    weekly["uplift_cm"] = (weekly["daydepth_m"] - ref) * 100.0
    if "botsflu_8wkrate" in sub:
        weekly["uplift_rate_cm_yr"] = (
            sub["botsflu_8wkrate"].resample("W-MON", label="left", closed="left").mean()
        )
    weekly.index.name = "week_start"
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(OUT_CSV)
    print(f"wrote {OUT_CSV}  ({len(weekly)} weeks; ref depth {ref:.3f} m)")
    print(weekly.describe())


if __name__ == "__main__":
    main()
