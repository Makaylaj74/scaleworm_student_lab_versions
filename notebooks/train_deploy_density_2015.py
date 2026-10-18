"""Train the deployable 2015-16 density model (option 2: two-model split).

The 5-fold CV established the honest held-out OOD performance (0.61x, r=0.60).
This trains ONE model on ALL 35 hand-counted 2015-16 frames (union with the
2021-24 clear set, heavy config -- there is no clear retention to protect in
this model; the separate clear model handles 2021-24), and saves its weights
for deployment on the 2015-16 era.

Two-model split:
  2021-24  -> csrnet_run/best.pt        (clear model, in-dist 0.96x / MAE 2.12)
  2015-16  -> best_ft2015.pt (this run) (fine-tuned, OOD ~0.61x per CV)

Cross-era scale: the two models have different bias vs GT, so to compare
ABSOLUTE counts across the 2017-20 gap, divide each block by its own ratio-to-
GT (2015-16 /0.61, clear /0.96). Within-block trends are unaffected.
"""

from __future__ import annotations

import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

import density_head as D
from finetune_density_2015_cv import OUT, SEED, load_gt, predict, train_fold

DEPLOY = OUT / "best_ft2015.pt"


def main() -> None:
    import random

    D.set_seed(SEED)
    random.seed(SEED)
    device = "cuda"
    items = load_gt()
    stems = [it[0] for it in items]
    gt = np.array([it[1] for it in items], float)

    log = (OUT / "deploy_train.log").open("w")
    print(f"training deploy 2015-16 model on ALL n={len(items)} frames", flush=True)
    model = train_fold(stems, device, log)
    torch.save(model.state_dict(), DEPLOY)

    # clear-val retention (expected to be degraded -- this model is NOT for clear)
    mae, _ = D.evaluate(
        model,
        DataLoader(D.WormDensity("val", False), batch_size=8, num_workers=6),
        device,
    )
    # in-sample 2015 fit (NOT a performance number -- training data; cite CV 0.61x)
    insample = predict(model, stems, device)
    ratio = insample.sum() / gt.sum()

    msg = (
        f"\nsaved deploy model -> {DEPLOY}\n"
        f"clear-val MAE {mae:.2f} (base 2.12; degraded as expected, "
        "clear model handles 2021-24)\n"
        f"in-sample 2015 ratio {ratio:.2f}x (training fit, NOT held-out; "
        "cite CV 0.61x for honest OOD performance)"
    )
    print(msg)
    log.write(msg + "\n")
    log.close()

    with (OUT / "deploy_insample_2015.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["stem", "gt", "insample_density"])
        for i, (st, g, _dc) in enumerate(items):
            wr.writerow([st, g, f"{insample[i]:.2f}"])


if __name__ == "__main__":
    main()
