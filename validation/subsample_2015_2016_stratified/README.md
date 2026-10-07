*AI-generated draft (Claude, Anthropic) — for review.*

# 2015-2016 density-stratified hand-count subsample

42 Scene-1 frames from the post-eruption era, picked to span detector-density terciles x time (sparse/moderate/dense, across 2015-2016). The detector is OUT-OF-DISTRIBUTION here (~28% recall) so `detector_count` is only a rough ordering used to stratify, NOT truth.

PURPOSE: (1) check whether the automated detector tracks anything in this era, (2) anchor the true level, (3) decide the measurement metric. Fill EITHER `worm_count` (individual count) OR `percent_cover` per the advisor's count-vs-cover decision (2015-2016 is blocked on this). Frames are gitignored (regenerable from this script).
