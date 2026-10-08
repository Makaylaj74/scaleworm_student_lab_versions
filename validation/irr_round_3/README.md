*AI-generated draft (Claude, Anthropic) — for review. Every step derives from the
version-controlled `scripts/build_irr_packet.py` and `scripts/irr_counts_agreement.py`.*

# IRR Round 3 — clean 3-way manual worm counts (MJ, Rater C, Rater B)

Round 2 compared independent worm **counts**, but it was **method-confounded**: MJ's
reference came from box-correcting detector *pre-labels* while Rater B counted *from scratch*,
so part of the ~40% gap was protocol, not rater. Round 3 fixes that: **all three raters
(MJ, MS, LG) count the same fresh frames blind, from scratch**, with no pre-labels for
anyone. Then we have a clean apples-to-apples 3-way agreement.

## The idea in one line

Big **images go out** (one-time, ~5 MB each), tiny **label text comes back** (a few
hundred KB) — so nothing needs a shared filesystem or git remote.

## The frame set (fresh — no overlap with round 2)

Clear-era Mondays only (no blur confound), spanning low to peak so we see whether
agreement degrades at high density. Peaks are pushed higher than round 2 (which maxed
~21/frame) because that is where the disagreement is expected to live.

| Period | Mondays | ~worms/frame | role |
|---|---|---|---|
| `low_2022` | 2022-02-14, 2022-04-11 | ~8 | low end |
| `peak_fall2022` | 2022-11-21, 2022-12-05, 2022-12-26 | ~30–37 | peak |

4 evenly-spaced 3-hourly slots per day → **20 frames** (8 low + 12 peak), the same split
as round 2. Regenerate / change via `ROUND`, `ROUNDS`, `--slots` in `build_irr_packet.py`.

## Workflow

1. **Build packets** (already done; regenerable):
   `python scripts/build_irr_packet.py MJ MS LG`
   → `validation/irr_round_3/packets/irr_packet_{MJ,MS,LG}.zip` (~95 MB each, gitignored).
2. **Send** `irr_packet_MS.zip` → Rater C, `irr_packet_LG.zip` → Rater B (email / Drive).
   **MJ does her own blind pass on the Hub:** `~/irr_packet_MJ` is symlinked to the
   packet, so open `~/irr_packet_MJ/labeler.ipynb` → **Restart & Run All** → click all 20
   frames **blind** (do not peek at the Monday series counts — that is what keeps MJ's
   pass from re-introducing the round-2 confound).
3. **Returns:** each rater sends back `<ID>_results.zip` (just `labels/` + `irr_manifest.csv`).
   Drop each into `validation/irr_round_3/raters/<ID>/` (unzipped).
4. **Score:**
   `python scripts/irr_counts_agreement.py validation/irr_round_3/raters/MJ \`
   `validation/irr_round_3/raters/MS validation/irr_round_3/raters/LG`
   → ICC(A,1) absolute agreement + per-pair |Δ|/bias/r (counts) and precision/recall/F1
   (boxes), each with bootstrap 95% CIs. Figure: `scripts/build_irr_figure.py`.

## One thing to confirm before packets go out

The rater README carries a **draft "what counts as one worm" rubric** (overlap/clumps,
edge-cut worms, tubes-vs-worms). Round 2 showed Rater B systematically under-counting, which a
shared, explicit rubric helps control — so it is worth MJ finalizing that wording once for
all three raters before sending, so any remaining disagreement is genuine rater variance,
not different instructions.
