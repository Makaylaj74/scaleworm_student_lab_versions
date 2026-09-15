"""Weekly Axial Seamount seismicity series (2021-2024) from the Wilcock catalog.

Reads the HYPO71-style full catalog (external_data/axial_eq_catalog/hypo71.dat;
see that dir's PROVENANCE.md for source + required citation), parses event
times + moment magnitudes, restricts to 2021-2024, and aggregates to weekly:
  - eq_count_all   : all located events per week
  - eq_count_mc    : events with MW >= Mc per week (Mc estimated from the data)
  - eq_max_mw      : largest MW that week

Mc (magnitude of completeness) is estimated by the maximum-curvature method on
the 2021-2024 magnitude-frequency distribution (peak of the non-cumulative
histogram, + a standard +0.2 bin correction) rather than carried over from
another study. The thresholded count is more comparable across time than raw
counts, whose smallest-event detectability drifts with noise and Navy
data-diversion gaps (see PROVENANCE.md caveats).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CAT = Path(__file__).resolve().parent.parent / (
    "external_data/axial_eq_catalog/hypo71.dat"
)
START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2025-01-01")
OUT_CSV = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / ("axial_seismicity_weekly_2017_2024.csv")
)
COLS = [
    "yyyymmdd",
    "hhmm",
    "sec",
    "lat_deg",
    "lat_min",
    "lon_deg",
    "lon_min",
    "depth",
    "mw",
    "nwr",
    "gap",
    "dmin",
    "rms",
    "erh",
    "erz",
    "id",
    "pmom",
    "smom",
]


def load_catalog(path: Path = CAT) -> pd.DataFrame:
    """Parse the HYPO71 catalog into a DataFrame with a UTC `time` column."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=1,
        names=COLS,
        dtype={"yyyymmdd": str, "hhmm": str},
        engine="python",
    )
    # zero-pad HHMM (e.g. "8" -> "0008"); build timestamp
    hhmm = df["hhmm"].str.zfill(4)
    hh = hhmm.str[:2].astype(int)
    mm = hhmm.str[2:].astype(int)
    base = pd.to_datetime(df["yyyymmdd"], format="%Y%m%d", errors="coerce")
    df["time"] = (
        base
        + pd.to_timedelta(hh, unit="h")
        + pd.to_timedelta(mm, unit="m")
        + pd.to_timedelta(df["sec"].astype(float), unit="s")
    )
    df = df.dropna(subset=["time"])
    return df


def estimate_mc(mw: pd.Series, bin_w: float = 0.1) -> float:
    """Maximum-curvature Mc: peak of the non-cumulative magnitude histogram
    plus a standard +0.2 correction (Wiemer & Wyss 2000)."""
    m = mw.dropna().to_numpy()
    edges = np.arange(np.floor(m.min()), np.ceil(m.max()) + bin_w, bin_w)
    counts, edges = np.histogram(m, bins=edges)
    peak = edges[:-1][np.argmax(counts)] + bin_w / 2.0
    return round(peak + 0.2, 2)


def weekly(df: pd.DataFrame, mc: float) -> pd.DataFrame:
    sub = df[(df["time"] >= START) & (df["time"] < END)].copy()
    g = sub.set_index("time")
    allc = g["id"].resample("W-MON", label="left", closed="left").count()
    mcc = g[g["mw"] >= mc]["id"].resample("W-MON", label="left", closed="left").count()
    maxmw = g["mw"].resample("W-MON", label="left", closed="left").max()
    out = pd.DataFrame({"eq_count_all": allc, "eq_count_mc": mcc, "eq_max_mw": maxmw})
    out["eq_count_mc"] = out["eq_count_mc"].fillna(0).astype(int)
    out.index.name = "week_start"
    return out


def main() -> None:
    df = load_catalog()
    print(
        f"catalog: {len(df):,} events "
        f"{df['time'].min().date()} .. {df['time'].max().date()}"
    )
    sub = df[(df["time"] >= START) & (df["time"] < END)]
    mc = estimate_mc(sub["mw"])
    print(f"2021-2024 events: {len(sub):,}  estimated Mc = {mc}")
    out = weekly(df, mc)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV)
    # stash Mc alongside for the plot/caption
    (OUT_CSV.with_suffix(".mc.txt")).write_text(f"{mc}\n")
    print(f"wrote {OUT_CSV}  ({len(out)} weeks)")
    print(out.describe())


if __name__ == "__main__":
    main()
