*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

# Axial Seamount earthquake catalog — provenance

**Purpose:** weekly seismicity time series (2021–2024) to graph alongside the
ASHES Mushroom thermistor record and the scaleworm abundance series.

## Source
- **File:** `hypo71.dat` (HYPO71-style full catalog of earthquake detections +
  HYPOINVERSE locations for the OOI cabled observatory at Axial Seamount).
- **URL:** http://axial.ocean.washington.edu/hypo71.dat
- **Maintainers:** William S. D. Wilcock & Maochuan Zhang, University of Washington.
- **Support:** U.S. National Science Foundation.
- **Downloaded:** 2026-09-14 (this session).
- **Server Last-Modified:** 2026-09-14 13:17:38 UTC (catalog updated hourly).
- **Size / checksum:** 36,479,380 bytes; md5 `f3a780faf725fa38a1b60949e3d9ea95`.
- **Coverage in file:** 2015-01-22 → 2026-09-14 (event IDs to ~571,164).

## Required citation (per the catalog site's "should be cited as")
- Wilcock, W. S. D., M. Tolstoy, F. Waldhauser, C. Garcia, Y. J. Tan,
  D. R. Bohnenstiehl, J. Caplan-Auerbach, R. P. Dziak, A. Arnulf, & M. E. Mann
  (2016). Seismic constraints on caldera dynamics from the 2015 Axial Seamount
  eruption, *Science*, 354, 1395–1399.
- Wilcock, W. S. D., F. Waldhauser, & M. Tolstoy (2017). Catalogs of earthquakes
  recorded on Axial Seamount... Interdisciplinary Earth Data Alliance (IEDA).
  https://doi.org/10.1594/IEDA/323843

## License / use
Public, NSF-supported near-real-time catalog with an explicit citation request
(no restrictive license stated). Use = cite the references above in any figure
caption, methods text, or presentation that uses these counts.

## Column layout (whitespace-delimited; 18 fields, header on line 1)
`yyyymmdd HHMM SS.SS lat_deg lat_min lon_deg lon_min depth MW NWR GAP DMIN RMS ERH ERZ ID PMom SMom`
- `MW` = moment magnitude (can be negative for microearthquakes).
- Longitudes are west (stored positive); site is ~45.95°N, 130.0°W.

## Caveats for a count time series (No Borrowed Assumptions)
- **Detection completeness varies:** the catalog notes intervals where the
  200 Hz / high-sample-rate data are **diverted by the U.S. Navy** and later
  backfilled — raw weekly counts can dip during un-backfilled diversions, an
  acquisition artifact rather than a true seismicity drop. Flag any anomalous
  low weeks against known diversion periods before interpreting.
- **Magnitude of completeness (Mc):** raw all-event counts are dominated by the
  smallest detectable events, whose detectability drifts with noise. A
  magnitude-thresholded count (e.g. MW ≥ Mc) is more stable across time; we
  emit both an all-event count and a thresholded count and state the threshold.
