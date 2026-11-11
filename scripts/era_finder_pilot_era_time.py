"""Pilot: density head read at the era's fixed Scene-1 time (fine 5 s grid t0-15..t0+30)."""

import sys
import time

sys.path.insert(0, "/home/jovyan/scaleworm-student-lab/notebooks")
from datetime import UTC, datetime

import build_worm_series_2021_2023_density as B
import cv2
import density_head as D
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

S = str(B.REPO / "validation/era_time_finder") + "/"


def era_t0(fid):
    day = fid[10:18]
    return 420 if day < "20210901" else 540 if day < "20220501" else 210


torch.set_num_threads(8)
model = D.CSRNet()
model.load_state_dict(torch.load(B.CLEAR_MODEL, map_location="cpu"))
model.eval()
norm = transforms.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)
ids = list(pd.read_csv(S + "diag_janaug_density.csv").frame_id)
s = pd.read_csv(str(B.REPO) + "/scene_sorting/monday_janaug_2021_3slot/sort_log.csv")
ids += list(s[s.decision == "not_scene1"].stem)
rows = []
t0c = time.time()
for fid in ids:
    dt = datetime.strptime(fid[10:], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    t0 = era_t0(fid)
    cap = cv2.VideoCapture(str(B.video_path(fid, dt)))
    for t in range(t0 - 15, t0 + 31, 5):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if ok:
            rows.append(
                (
                    fid,
                    t0,
                    t,
                    B.density_count(
                        model,
                        Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)),
                        "cpu",
                        norm,
                    )
                    / B.RATIO,
                )
            )
    cap.release()
print(f"{len(ids)} recs, {len(rows)} frames in {time.time() - t0c:.0f}s")
pd.DataFrame(rows, columns=["frame_id", "t0", "t", "dh"]).to_csv(
    S + "pilot_era_time.csv", index=False
)
