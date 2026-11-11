"""Era-fixed Scene-1-time frame finder, validation step (Nov 2026; see method_writeup_detector_2026_10_14.md §5).

Outputs go to validation/era_time_finder/. Run from repo root with OMP_NUM_THREADS=8 (CPU).
"""
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

torch.set_num_threads(8)
model = D.CSRNet()
model.load_state_dict(torch.load(B.CLEAR_MODEL, map_location="cpu"))
model.eval()
norm = transforms.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)
d = pd.read_csv(str(B.REPO / "validation/era_time_finder") + "/diag_janaug_density.csv")
rows = []
t0 = time.time()
for fid in d.frame_id:
    dt = datetime.strptime(fid[10:], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    cap = cv2.VideoCapture(str(B.video_path(fid, dt)))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    for t in range(30, int(dur), 60):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if ok:
            c = (
                B.density_count(
                    model,
                    Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)),
                    "cpu",
                    norm,
                )
                / B.RATIO
            )
            rows.append((fid, t, c))
    cap.release()
print(f"{len(rows)} frames in {time.time() - t0:.0f}s")
pd.DataFrame(rows, columns=["frame_id", "t", "dh"]).to_csv(
    str(B.REPO / "validation/era_time_finder") + "/pilot_dh_grid.csv", index=False
)
