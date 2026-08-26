"""Single-frame extraction and YOLO worm-counting helpers.

Shared by the population-count notebook (``24_count_timeseries``) and the model
comparison notebook (``26_compare_v0_v1``). Keeps the pure, testable logic —
stem→path mapping, timestamp parsing, frame extraction, and box counting — out of
the notebooks so it can be unit-tested and reused.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


def stem_to_video(stem: str, video_root: Path) -> Path:
    """Map a CAMHD stem to its video file.

    ``CAMHDA301-YYYYMMDDThhmmss`` → ``<video_root>/YYYY/MM/DD/<stem>.mp4``.
    """
    dt = stem.split("-")[1]  # e.g. '20240205T001500'
    return video_root / dt[0:4] / dt[4:6] / dt[6:8] / f"{stem}.mp4"


def parse_stem_dt(stem: str) -> datetime:
    """Return the recording datetime encoded in a CAMHD stem."""
    return datetime.strptime(stem.split("-")[1], "%Y%m%dT%H%M%S")


def extract_frame(
    video_path: Path, t_s: float, out_png: Path, timeout: int = 120
) -> bool:
    """Extract one full-resolution frame at ``t_s`` seconds to ``out_png``.

    Uses ``-ss`` before ``-i`` for a fast (keyframe) seek — adequate given the
    ~30 s precision of the contact-sheet-derived timestamps. Returns whether the
    frame was written. No downscaling: the frame is full resolution for detection.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(t_s),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_png),
        ],
        capture_output=True,
        timeout=timeout,
    )
    return out_png.exists()


def count_worms(
    box_class_ids: list[int],
    names: dict[int, str],
    worm_classes: frozenset[str] = frozenset({"scale_worm"}),
) -> int:
    """Count detection boxes that represent worms.

    A single-class model (e.g. ``{0: 'scale_worm'}``) counts every box. A
    multi-class model counts only boxes whose class name is in ``worm_classes``
    (case-insensitive *exact* match — not substring, so a ``not_worm`` class is
    never miscounted as a worm). This keeps the code correct whether the chosen
    model (mushroom.pt / v1 / v2) is single- or multi-class.
    """
    single_class = len(names) == 1
    targets = {c.lower() for c in worm_classes}
    return sum(
        1 for cid in box_class_ids if single_class or names[cid].lower() in targets
    )
