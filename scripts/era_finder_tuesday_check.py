"""Independent Tuesday check of the era-fixed Scene-1-time density read (rules fixed from Mondays)."""

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


def era_times(day: str) -> tuple[int, ...]:
    """Scene-1 time(s) per camera-schedule era, derived from the MONDAY series only."""
    if day < "20210901":
        return (420,)
    if day < "20220425":
        return (540,)
    if "20230101" <= day < "20230601":
        return (210, 420)
    return (210,)


torch.set_num_threads(8)
model = D.CSRNet()
model.load_state_dict(torch.load(B.CLEAR_MODEL, map_location="cpu"))
model.eval()
norm = transforms.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)
t = pd.read_csv(str(B.REPO) + "/scene_sorting/tuesdays_2021_2024/sort_log.csv")
t = t[t.stem.str[10:18] < "20230810"]
rows = []
t0c = time.time()
for fid in t.stem:
    dt = datetime.strptime(fid[10:], "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    cap = cv2.VideoCapture(str(B.video_path(fid, dt)))
    for tt in era_times(fid[10:18]):
        cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000)
        ok, fr = cap.read()
        if ok:
            rows.append(
                (
                    fid,
                    tt,
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
print(f"{len(t)} recs, {len(rows)} reads in {time.time() - t0c:.0f}s")
pd.DataFrame(rows, columns=["frame_id", "t", "dh"]).to_csv(
    S + "tuesday_check.csv", index=False
)
