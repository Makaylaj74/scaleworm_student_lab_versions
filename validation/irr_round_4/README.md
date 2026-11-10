*AI-generated draft (Claude, Anthropic) — for review. Every step derives from the
version-controlled `scripts/build_irr_triplet_packet.py` and `scripts/irr_counts_agreement.py`.*

# IRR Round 4: blind counts on before / at / after triplets (Rater B, Rater C)

Katie recommended that every frame a rater annotates come with the frame just **before**
and just **after** it. With those neighbours, a rater can tell whether a boxed shape is a
**shadow** (it stays fixed while lighting and edges change) or a **worm** (it shifts or
bends). Round 4 does this: **5 moments × 3 frames (t − 10 s, t, t + 10 s) = 15 frames**.
Every frame is annotated blind and from scratch. The frames are shown grouped by triplet,
and the labeler has a **View time** switch that flips to the other two frames of the
triplet with the current boxes drawn on top. Clicking is turned off while a neighbour is
showing.

## The frame set

Clear-era (2021-09 to 2023-08) counted Monday frames. None of their days were used in
rounds 2 or 3. Each comes from a different month and sits nearest one of the
10/30/50/70/90th percentiles of MJ's Monday count, so the set covers low to high density.

| Triplet | Anchor | MJ Monday count | sharpness −10/0/+10 (ratio) |
|---|---|---|---|
| T1 | 2023-06-26 03:15 | 7 | 87/113/107 (0.77) |
| T2 | 2023-03-27 09:15 | 10 | 194/219/234 (0.83) |
| T3 | 2023-08-21 15:15 | 13 | 65/79/116 (0.57) |
| T4 | 2022-09-19 21:15 | 17 | 137/185/163 (0.74) |
| T5 | 2022-08-29 21:15 | 25 | 88/134/160 (0.55) |

The top of the range is about 25 worms/frame. That is lower than round 3's fall-2022 peaks
(about 30–47), where the disagreement was largest. If the shadow/movement question matters
most at peak density, a denser triplet can be swapped in (change `QUANTILES`/`ANCHORS`).

**Triplet QC.** This is enforced on every build, and `--pick` keeps the first passing
candidate nearest each percentile:

- **Same view:** each neighbour must register to its anchor by phase correlation, with a
  shift of at most 10 px. Probe: on 2022-11-07 12:15 the camera held the identical framing
  for at least ±20 s.
- **Matched focus:** the triplet's min/max sharpness (median 3×3-tile Laplacian variance)
  must be at least 0.5. This threshold was set by eye on the first 5 candidates. The two
  that looked matched scored 0.77 and 0.55; the three with a visibly softer member scored
  0.22–0.43.

Focus drift within ±10 s is common in this record: **8 of 13 candidates failed**. Two
failed on a transient refocus or camera move, and six had a softer neighbour.

## Round trip

```bash
python scripts/build_irr_triplet_packet.py LG MS     # -> packets/irr_packet_{LG,MS}_round4.zip
```

1. Send `irr_packet_LG_round4.zip` to Rater B and `irr_packet_MS_round4.zip` to Rater C.
   The packet folder is named `irr_packet_<ID>_round4`, so it can't be confused with an
   older `irr_packet_<ID>` still in their home folder. That confusion is what sent
   Rater C's round-3 upload to the wrong packet.
2. Raters return `<ID>_results.zip` (`labels/` + `irr_manifest.csv`). Unzip it into
   `raters/<ID>/`.
3. Score with `python scripts/irr_counts_agreement.py raters/LG raters/MS`.

**Scoring caveat:** the 3 frames of a triplet are not independent. The current
`irr_counts_agreement.py` bootstraps per frame, so its CIs will be too narrow here.
Resample **by triplet** (cluster bootstrap) before reporting CIs. The extra triplet
information also enables new measures: within-rater count stability across t−10/t/t+10,
and whether a box persists across the triplet (worm) or not (possible shadow).
