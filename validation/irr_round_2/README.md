*AI-generated draft (Claude, Anthropic) — for review. Every step derives from the
version-controlled `scripts/build_irr_packet.py` and `scripts/irr_counts_agreement.py`.*

# IRR Round 2 — manual worm counts (Lyr & Maureen)

Round 1 validated the **scene-sorting** decision (Scene-1 vs not) with kappa. Round 2
validates the **counting** step: do independent raters get the same worm *counts* (and
mark the same worms) on the manual annotation frames? Because the two containers are
not directly connected, this runs as a **portable-packet** round-trip — the same shape
as Round 1's rater zips, but for boxes/counts instead of categories.

## The idea in one line

Big **images go out** (one-time, ~5 MB each), tiny **label text comes back** (a few
hundred KB) — so nothing needs a shared filesystem or git remote.

## The frame set

Chosen with Makayla: a low-abundance period vs the peak, all from Mondays that are
actually sorted+counted in `validation/monday_manual_series/`.

| Period | Mondays | role |
|---|---|---|
| `low_jan2022` | 2022-01-17, 2022-01-31 | low end |
| `peak_jan2023` | 2023-01-09, 2023-01-16, 2023-01-23 | peak |

Up to `SLOTS_PER_DAY` (default 4) evenly-spaced 3-hourly slots per day. Edit `PERIODS`
/ `--slots` in `build_irr_packet.py` to change the set. (Jan 2022 only has 2 fully
counted Mondays in the series — there is no third, so the low end is 2 days by 3.)

## Blind, from-scratch — and why

Each rater draws every worm on an **empty** frame (no pre-labels). This measures
*independent* reliability. Shipping the v2/v3 pre-labels for them to "correct" would
inflate agreement through shared anchoring — that would answer a different question
(`from_scratch` vs `box_corrected` in the schema).

⚠️ **Protocol note for the third rater (Makayla).** The production series was built by
box-*correcting* pre-labels (`annotation_method=box_corrected`). For a clean 3-way
*blind* comparison, Makayla should also do a blind pass on these frames via her own
`irr_packet_MJ` — do **not** just reuse the production `labels/`, or MJ's protocol
differs from LG/MS and the IRR is confounded. (If you accept that caveat and prefer to
use the production labels, point the agreement script at a folder holding those labels +
a manifest instead.)

## Workflow

### 1. Build the packets (Makayla, on the Hub)
```bash
cd /home/jovyan/scaleworm-student-lab
python scripts/build_irr_packet.py LG MS MJ
# -> validation/irr_round_2/packets/irr_packet_{LG,MS,MJ}.zip
```

### 2. Send the packets out
Give `irr_packet_LG.zip` to Lyr and `irr_packet_MS.zip` to Maureen (email or a shared
Drive folder — each is ~tens of MB of images). Makayla keeps/annotates `irr_packet_MJ`.

### 3. They annotate (each rater, independently)
Instructions are inside each zip (`README.md`). In short: unzip into the home directory,
open `labeler.ipynb` (uses the generic **Python 3** kernel — the first cell installs
`ipympl` into it, so no special or shared kernel is needed), Restart & Run All,
left-click every worm, **Save & Next** through all frames.

### 4. They send results back
Each rater returns **only** `labels/` + `irr_manifest.csv` (a `<id>_results.zip`, a few
hundred KB). The images do **not** come back.

### 5. Import + score (Makayla)
Drop each returned zip's contents into its rater folder:
```
validation/irr_round_2/raters/LG/{irr_manifest.csv, labels/}
validation/irr_round_2/raters/MS/{irr_manifest.csv, labels/}
validation/irr_round_2/raters/MJ/{irr_manifest.csv, labels/}   # Makayla's blind pass
```
Then:
```bash
python scripts/irr_counts_agreement.py \
    validation/irr_round_2/raters/MJ \
    validation/irr_round_2/raters/LG \
    validation/irr_round_2/raters/MS
```

## What you get

Only frames **every** rater marked `counted` enter the analysis (excluded frames are
listed, so coverage is explicit). All statistics carry a bootstrap 95% CI.

- **Count level** — ICC(A,1) absolute agreement across all raters; per pair: mean
  absolute count difference, bias, Pearson r, exact-count agreement.
- **Box level** — per pair, a centre-in-box match → precision / recall / F1 (do they
  mark the *same* worms, not just the same total?).

Two frames disagreeing wildly are worth eyeballing together — that is usually where a
counting-rule ambiguity (what counts as one worm, tubes vs bodies, edge worms) hides,
which is the real payoff of running IRR before trusting the series.
