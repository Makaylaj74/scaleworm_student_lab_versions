"""Era-fixed Scene-1-time frame finder, validation step (Nov 2026; see method_writeup_detector_2026_10_14.md §5).

Outputs go to validation/era_time_finder/. Run from repo root with OMP_NUM_THREADS=8 (CPU).
"""
import sys

sys.path.insert(0, "/home/jovyan/scaleworm-student-lab/notebooks")
import build_worm_series_2021_2023_density as B
import density_head as D
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

R = B.REPO / "validation"
m = pd.concat(
    [
        pd.read_csv(R / f / "frame_manifest.csv").assign(src=f)
        for f in (
            "recount_janaug_2021",
            "recount_janaug_2021_rest",
            "recount_sepdec_2021_rest",
        )
    ]
)
m = m[m.frame_status == "counted"].copy()
torch.set_num_threads(8)
model = D.CSRNet()
model.load_state_dict(torch.load(B.CLEAR_MODEL, map_location="cpu"))
model.eval()
norm = transforms.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)
m["dh_on_mj_frame"] = [
    B.density_count(
        model, Image.open(R / s / "images" / f"{fid}.png").convert("RGB"), "cpu", norm
    )
    / B.RATIO
    for s, fid in zip(m.src, m.frame_id)
]
pf = pd.read_csv(B.PERFRAME).set_index("frame_id")
h = (
    pd.concat([pd.read_csv(x) for x in B.HARVESTS])
    .drop_duplicates("frame_id")
    .set_index("frame_id")
)
m["dh_on_harvest_frame"] = m.frame_id.map(pf.worm_index)
m["harvest_t"] = m.frame_id.map(h.best_time_s)
m.to_csv(
    str(B.REPO / "validation/era_time_finder") + "/diag_janaug_density.csv", index=False
)
m["period"] = np.where(m.frame_id.str[10:16] < "202109", "JanAug", "SepDec")
m["n"] = m.worm_count.astype(float)
for p, g in m.groupby("period"):
    gg = g.dropna(subset=["dh_on_harvest_frame"])
    print(
        p,
        "n",
        len(g),
        "hand",
        round(g.n.mean(), 1),
        "| DH on MJ frame",
        round(g.dh_on_mj_frame.mean(), 1),
        "ratio",
        round(g.dh_on_mj_frame.sum() / g.n.sum(), 2),
        "| n w/ harvest",
        len(gg),
        "DH on harvest frame",
        round(gg.dh_on_harvest_frame.mean(), 1),
        "ratio",
        round(gg.dh_on_harvest_frame.sum() / gg.n.sum(), 2),
        "| harvest_t==MJ_t",
        int((abs(gg.harvest_t - gg.scene1_time_s) <= 30).sum()),
        "/",
        len(gg),
    )
    print(
        "   harvest t distribution:", gg.harvest_t.value_counts().sort_index().to_dict()
    )
