*AI-generated draft (Claude, Anthropic) — for review. Every step derives from the
version-controlled `scripts/build_irr_triplet_packet.py` and `scripts/irr_counts_agreement.py`.*

# IRR Round 4: blind counts on before / at / after triplets (MJ, Rater B, Rater C)

Katie recommended that every frame a rater annotates come with the frame just **before**
and just **after** it. With those neighbours, a rater can tell whether a boxed shape is a
**shadow** (it stays fixed while lighting and edges change) or a **worm** (it shifts or
bends). Round 4 does this: **5 moments × 3 frames (t − 10 s, t, t + 10 s) = 15 frames**.
Every frame is annotated blind and from scratch. The frames are shown grouped by triplet,
and the labeler has a **View time** switch that flips to the other two frames of the
triplet with the current boxes drawn on top. Clicking is turned off while a neighbour is
showing.

## The frame set: clean for every rater, MJ included

The first draft used counted Monday frames. MJ had box-corrected those before, so they
were not blind for her. Round 4 instead uses **clear-era (2021-09 to 2023-08) Tue–Fri
recordings on days where no recording has ever been hand-counted, box-labelled or
scene-sorted by anyone**. Any CAMHD stem found under `validation/`, `datasets/`,
`scene_sorting/` or the hand-count/IRR folders in `notebooks/` blocks its whole day.
There is one per calendar month. Density is spread using the automated v8+SAHI harvest
count, taking the recordings nearest the 10/30/50/70/90th percentiles. That count is used
only for selection: it is never written into a packet, a sheet or the notebook, and never
printed.

**Scene-1 needs a human eye.** The harvest's auto-picked frame time often lands off
Scene-1: 2 of the first 5 candidates were tubeworm views. Image matching against a Monday
Scene-1 frame can't separate them either, because true Scene-1 frames a week apart score
0.02–0.04, as low as off-scene views. So MJ does a **scene-only** yes/no + time call on
20 candidate contact sheets (4 per density level) in
`notebooks/41_scene_check_irr_round4.ipynb`. No counts are shown, and this is the same
scene-sort step every earlier round had. `--pick` then keeps, per level, the first
Scene-1 sheet whose triplet passes QC.

## Round trip

```bash
python scripts/build_irr_triplet_packet.py --scene-sheets   # 20 candidate sheets
#   MJ scene-checks them in notebooks/41_scene_check_irr_round4.ipynb
python scripts/build_irr_triplet_packet.py --pick           # paste result into ANCHORS
python scripts/build_irr_triplet_packet.py MJ LG MS         # -> packets/irr_packet_*_round4.zip
```

1. MJ annotates her own `irr_packet_MJ_round4` blind. Send `irr_packet_LG_round4.zip` to Rater B and `irr_packet_MS_round4.zip` to Rater C.
   The packet folder is named `irr_packet_<ID>_round4`, so it can't be confused with an
   older `irr_packet_<ID>` still in their home folder. That confusion is what sent
   Rater C's round-3 upload to the wrong packet.
2. Raters return `<ID>_results.zip` (`labels/` + `irr_manifest.csv`). Unzip it into
   `raters/<ID>/`.
3. Score with `python scripts/irr_counts_agreement.py raters/MJ raters/LG raters/MS`.

**Scoring caveat:** the 3 frames of a triplet are not independent. The current
`irr_counts_agreement.py` bootstraps per frame, so its CIs will be too narrow here.
Resample **by triplet** (cluster bootstrap) before reporting CIs. The extra triplet
information also enables new measures: within-rater count stability across t−10/t/t+10,
and whether a box persists across the triplet (worm) or not (possible shadow).
