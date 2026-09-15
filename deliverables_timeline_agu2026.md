*AI-generated draft (Claude, Anthropic) — for review. All parameters, series, and figures are derived from version-controlled scripts and data. Edit freely; this is a planning document, not a result.*

# Scaleworm × Axial geophysics — deliverables timeline (AGU 2026)

**Goal:** a **2017–2024** scaleworm population series vs Axial Seamount geophysical
signals (diffuse-flow temperature, caldera inflation, seismicity), plus a
**dw/dt** (rate-of-change) series, ready by **mid-November 2026** for the AGU
poster (AGU early December).

**Owner:** Makayla Joseph. **Advisor:** Dax Soule. **Students:** Lyr Gada
(2017–2018 scene sort), + one to assign (2019–2020 scene sort), Maureen Scully
(OOI 2017–2018 camera timeline).

## Method model (why this is feasible in 2 months)
- **Manual = ground-truth baseline** — box-corrected counts, dense (Monday × 8
  slots). Covers **2021–2024** (Makayla). This anchors and validates everything.
- **AI = scalable historical counting** — the v2 (clear) / v3 (blurry) detector
  runs on each sorted Scene-1 frame. Covers **2017–2020**. Every camera-year is
  validated by a small **per-unit hand-count gate** (~20–30 frames) so its recall
  is known; a unit that fails its gate is caveated or excluded, never faked.
- **Scene-sorting stays manual** (an AI sorter was tested and failed, κ = −0.42),
  but it is **distributed across people** so no one carries the whole load.

## Coverage & ownership

| Period | Scene sort | Count | Owner / status |
|---|---|---|---|
| 2023–2024 | done | manual | done (367 frames) |
| Sep 2021 – Dec 2022 | done | manual | Makayla — nb 33 (351 pending) |
| Spring 2021 (temp-spike) | in progress | manual | Makayla — nb 25 → nb 33 |
| **2019–2020** | **to assign** | AI + gate | **new student** (assignment to draft) |
| 2017–2018 | in progress | AI + gate | **Lyr Gada** |

Camera units swap each August, so 2017–2024 spans ~8 units; each AI-counted unit
needs its own recall gate. **2018 is ~half black footage** (lights off) → expect a
thin year. 2015–2016 = optional stretch only if time allows.

## Timeline (mid-Sep → mid-Nov)

**Phase A — Foundations (Sep 15–26)**
- [Claude] Sensor series extended to **2017–2024** (temperature, inflation,
  seismicity) — DONE/in progress. Build the **AI counting pipeline**
  (`run_ai_counts.py`) and the **manual-vs-AI** comparison. — in progress.
- [Makayla] Finish the 351 pending 2021–2022 counts (nb 33); sort the spring-2021
  spike weeks (nb 25). Assign the 2019–2020 sorter.
- [Lyr] Sort 2017–2018.

**Phase B — Historical ingestion (Sep 29 – Oct 17)**
- [Students] Sort 2017–2020.
- [Makayla/Lyr] Hand-count **per-unit gate** frames (~20–30 per camera-year,
  2016–2020) to measure detector recall.
- [Claude] AI-count each year as its sort lands; run manual-vs-AI on 2021–2024;
  report which units pass/fail their gate (decide include / caveat / quick fine-tune).

**Phase C — Assembly (Oct 20 – Nov 7)**
- [Claude] Stitch the full **2017–2024 population series** (manual baseline + AI
  extension, confidence-tiered by per-unit recall). Compute **dw/dt** (smoothed,
  bootstrap 95% CI). Build population-vs-sensor and dw/dt figures.

**Phase D — Polish + stats (Nov 10–14)**
- [Claude] Lead-lag / correlation between dw/dt and inflation / seismicity /
  temperature. Poster-ready figures. Buffer for slippage.
- [Makayla + Dax] Review and sign-off.

**→ Mid-November: analysis + figures complete. Late Nov: build AGU poster.**

## Key risks (both surface by mid-October — early enough to react)
1. **Detector recall on pre-2021 camera units.** v2/v3 were tuned on −2021/−2022;
   older units may recall poorly (−2021 was 37%). The per-unit gates reveal this in
   Phase B; low-recall units get a recall-correction + CI, a quick fine-tune, or a
   low-confidence flag.
2. **Student sorting throughput.** If a sorter runs slow, drop that year to coarse
   density (2–4 recordings/month) — still a valid trend + dw/dt, fewer points.
3. **2021–2024 baseline must be reasonably complete by mid-Oct** — it is what
   justifies trusting the AI counts for 2017–2020.

## Definitions
- **dw/dt** — weekly rate of change of worm abundance (finite difference of the
  population series, smoothed), with bootstrap CI.
- **Bootstrap 95% CI** — uncertainty from resampling the data ~10⁴ times with
  replacement and taking the 2.5–97.5 percentile; assumes no particular distribution.
- **Lead-lag** — sliding one series by k weeks vs another to find the offset of
  peak correlation; evidence a signal *precedes* (drives) the other.
