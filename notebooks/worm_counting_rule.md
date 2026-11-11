*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

# What counts as one worm

**Status:** confirmed by Makayla Joseph, 2026-11-11. Applies to every manual box count:
the Monday series, the recount packets, and the IRR packets. Extends the IRR draft rubric
in `scripts/build_irr_packet.py` (≈ line 488) with an explicit dark-opening rule.

- Box each **individual worm body you can make out**, one box per worm.
- If worms overlap or clump, count as many **distinct** ones as you can separate by eye;
  don't guess at ones you can't actually distinguish.
- A worm **half-cut-off at the edge** still counts if you can tell it's a worm.
- Count the **worms**, not the white tubes or background texture.
- **Dark openings/holes:** count only if part of the worm's body (reddish-brown,
  segmented, or scaled) is visible inside it. A dark opening with no visible body is
  not a worm.
- Don't count the same worm twice. Be consistent with yourself across all frames; take
  breaks rather than counting while tired.

## Why the dark-opening rule was added

On 2026-11-11 the blind Sep–Dec 2021 recount showed the counting criterion had shifted
within one day. Frames from the same 13 Mondays, which agreed in the original
September 2026 counts (ratio 1.05), differed by 2.35× between the first recount (dark
openings boxed) and the later one (dark openings boxed only when a body was visible).
Counts made under the broad criterion (the Jan–Aug 2021 set in
`validation/monday_janaug_2021/` and the 15-frame sample in
`validation/recount_sepdec_2021/`) are **not comparable** with this rule and need
recounting before use.
