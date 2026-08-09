"""Sample CAMHD recordings on a fixed cadence and build contact sheets for scene sorting.

The Mushroom-vent CAMHD camera runs a scripted pan/tilt/zoom program inside each
~15-minute recording, and the timing of that program drifts across days-to-weeks.
That drift makes a single fixed frame time unreliable for locating "Scene 1" (the
front-on Mushroom view used for scale-worm counts). This module does not try to
find Scene 1 automatically. Instead it samples recordings on a defensible, regular
cadence (e.g. every Monday, all eight 3-hourly slots) and renders a labeled
*contact sheet* per recording — a grid of frames spanning the whole clip — so a
human can see the full PTZ program and sort each recording into Scene 1 / not.

Layout of the OOI archive (UTC throughout)::

    <base>/YYYY/MM/DD/CAMHDA301-YYYYMMDDTHHMMSS.mp4

Eight recordings per day at a 3-hour cadence: 00:15, 03:15, 06:15, 09:15, 12:15,
15:15, 18:15, 21:15 UTC.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on the Hub
import matplotlib.pyplot as plt

# --- Archive conventions -----------------------------------------------------

INSTRUMENT = "CAMHDA301"
#: The eight UTC recording slots per day, as HHMMSS strings (3-hour cadence).
SLOTS: tuple[str, ...] = (
    "001500",
    "031500",
    "061500",
    "091500",
    "121500",
    "151500",
    "181500",
    "211500",
)

#: Worker ceiling for this host (JupyterHub container is CPU-capped). Never
#: exceed this without explicit instruction — see lab CLAUDE.md.
MAX_WORKERS = 24


# --- Pure logic (unit-tested) ------------------------------------------------


def iter_sample_dates(
    start: date, end: date, weekday: int = 0, step_weeks: int = 1
) -> list[date]:
    """Return sampled dates from ``start`` to ``end`` inclusive.

    Picks the given ``weekday`` (Mon=0 ... Sun=6) and steps by ``step_weeks``
    weeks. The first sampled date is the first occurrence of ``weekday`` on or
    after ``start``.

    Args:
        start: First calendar date to consider (inclusive).
        end: Last calendar date to consider (inclusive).
        weekday: Target weekday, Monday=0 ... Sunday=6.
        step_weeks: Weeks between samples (1 = weekly, 4 ≈ ~monthly).

    Returns:
        Ordered list of dates, possibly empty.
    """
    if step_weeks < 1:
        raise ValueError("step_weeks must be >= 1")
    if not 0 <= weekday <= 6:
        raise ValueError("weekday must be in 0..6 (Mon..Sun)")

    offset = (weekday - start.weekday()) % 7
    current = start + timedelta(days=offset)
    stride = timedelta(weeks=step_weeks)
    dates: list[date] = []
    while current <= end:
        dates.append(current)
        current += stride
    return dates


def recording_datetime(day: date, slot: str) -> datetime:
    """Combine a date and an ``HHMMSS`` slot string into a UTC datetime."""
    if len(slot) != 6 or not slot.isdigit():
        raise ValueError(f"slot must be a 6-digit HHMMSS string, got {slot!r}")
    hh, mm, ss = int(slot[0:2]), int(slot[2:4]), int(slot[4:6])
    return datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=UTC)


def recording_stem(dt: datetime) -> str:
    """Return the archive filename stem, e.g. ``CAMHDA301-20241017T121500``."""
    return f"{INSTRUMENT}-{dt:%Y%m%dT%H%M%S}"


def recording_path(base: Path, dt: datetime) -> Path:
    """Return the expected mp4 path for a recording datetime under ``base``."""
    return base / f"{dt:%Y}" / f"{dt:%m}" / f"{dt:%d}" / f"{recording_stem(dt)}.mp4"


def frame_offsets(duration_s: float, every_s: float, pad_s: float = 1.0) -> list[float]:
    """Return frame sample offsets (seconds) spanning a clip of ``duration_s``.

    Starts at 0 and steps by ``every_s`` up to ``duration_s - pad_s`` (the pad
    avoids seeking past the last decodable frame).

    Args:
        duration_s: Clip duration in seconds.
        every_s: Spacing between sampled frames in seconds.
        pad_s: Tail margin excluded from sampling.
    """
    if every_s <= 0:
        raise ValueError("every_s must be > 0")
    if duration_s <= 0:
        return []
    limit = max(0.0, duration_s - pad_s)
    offsets: list[float] = []
    t = 0.0
    while t <= limit:
        offsets.append(round(t, 3))
        t += every_s
    return offsets


def grid_shape(n: int, cols: int) -> tuple[int, int]:
    """Return ``(rows, cols)`` needed to lay out ``n`` tiles in ``cols`` columns."""
    if cols < 1:
        raise ValueError("cols must be >= 1")
    if n <= 0:
        return (0, cols)
    rows = -(-n // cols)  # ceil division
    return (rows, cols)


# --- Target enumeration ------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """A single recording we intend to sample."""

    dt: datetime
    path: Path

    @property
    def stem(self) -> str:
        return recording_stem(self.dt)

    @property
    def exists(self) -> bool:
        return self.path.is_file()


def enumerate_targets(
    base: Path, dates: Iterable[date], slots: Sequence[str] = SLOTS
) -> list[Target]:
    """Build the full list of intended recordings (whether or not they exist)."""
    targets: list[Target] = []
    for day in dates:
        for slot in slots:
            dt = recording_datetime(day, slot)
            targets.append(Target(dt=dt, path=recording_path(base, dt)))
    return targets


# --- ffmpeg / ffprobe wrappers -----------------------------------------------


def probe_duration(video: Path) -> float | None:
    """Return the video duration in seconds via ffprobe, or None on failure."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def extract_frame(video: Path, t: float, out_png: Path, thumb_w: int = 480) -> bool:
    """Extract one frame at offset ``t`` seconds to ``out_png`` (downscaled).

    Uses input seeking (``-ss`` before ``-i``) for speed. Returns True on success.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        f"scale={thumb_w}:-1",
        "-q:v",
        "3",
        "-y",
        str(out_png),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return out_png.is_file()
    except subprocess.CalledProcessError:
        return False


def build_contact_sheet(
    video: Path,
    out_png: Path,
    every_s: float = 30.0,
    cols: int = 6,
    thumb_w: int = 480,
    dpi: int = 150,
) -> Path | None:
    """Render a labeled contact sheet for one recording.

    Extracts a frame every ``every_s`` seconds across the clip, lays them out in a
    grid, labels each tile with its offset, and titles the sheet with the recording
    stem. Returns the output path, or None if the clip could not be read.
    """
    duration = probe_duration(video)
    if duration is None:
        return None
    offsets = frame_offsets(duration, every_s)
    if not offsets:
        return None

    rows, cols = grid_shape(len(offsets), cols)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        images: list[tuple[float, Path]] = []
        for i, t in enumerate(offsets):
            frame_png = tmp_dir / f"f{i:03d}.png"
            if extract_frame(video, t, frame_png, thumb_w=thumb_w):
                images.append((t, frame_png))
        if not images:
            return None

        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 2.4, rows * 1.6), squeeze=False
        )
        for ax in axes.flat:
            ax.axis("off")
        for idx, (t, frame_png) in enumerate(images):
            ax = axes[idx // cols][idx % cols]
            ax.imshow(plt.imread(frame_png))
            ax.set_title(f"t={t:.0f}s", fontsize=7, pad=2)
        fig.suptitle(video.stem, fontsize=10, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)

    return out_png


# --- Orchestration -----------------------------------------------------------


def clamp_workers(requested: int) -> int:
    """Clamp the worker count to the host ceiling, warning if it was too high."""
    if requested > MAX_WORKERS:
        print(
            f"[scene_sampler] requested {requested} workers; "
            f"clamping to host ceiling {MAX_WORKERS}",
            file=sys.stderr,
        )
        return MAX_WORKERS
    return max(1, requested)


def write_manifest(targets: Sequence[Target], sheets_dir: Path, out_csv: Path) -> None:
    """Write a CSV row per target: date, slot, existence, and sheet path."""
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["stem", "utc_datetime", "weekday", "slot", "video_exists", "sheet_path"]
        )
        for t in targets:
            sheet = sheets_dir / f"{t.stem}.png"
            writer.writerow(
                [
                    t.stem,
                    t.dt.isoformat(),
                    t.dt.strftime("%A"),
                    t.dt.strftime("%H%M%S"),
                    t.exists,
                    str(sheet) if sheet.is_file() else "",
                ]
            )


def run(
    base: Path,
    out_dir: Path,
    start: date,
    end: date,
    weekday: int = 0,
    step_weeks: int = 1,
    slots: Sequence[str] = SLOTS,
    every_s: float = 30.0,
    cols: int = 6,
    workers: int = 8,
    limit: int | None = None,
    dry_run: bool = False,
) -> list[Target]:
    """Sample recordings and build contact sheets. Returns the target list.

    Creates ``out_dir/{contact_sheets,scene1,not_scene1}`` and a manifest CSV.
    """
    dates = iter_sample_dates(start, end, weekday=weekday, step_weeks=step_weeks)
    targets = enumerate_targets(base, dates, slots=slots)
    present = [t for t in targets if t.exists]

    sheets_dir = out_dir / "contact_sheets"
    for sub in ("contact_sheets", "scene1", "not_scene1"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    print(
        f"[scene_sampler] {len(dates)} sampled days x {len(slots)} slots "
        f"= {len(targets)} target recordings; {len(present)} present on disk"
    )
    if dry_run:
        write_manifest(targets, sheets_dir, out_dir / "manifest.csv")
        return targets

    todo = present if limit is None else present[:limit]
    workers = clamp_workers(workers)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                build_contact_sheet,
                t.path,
                sheets_dir / f"{t.stem}.png",
                every_s,
                cols,
            ): t
            for t in todo
        }
        for fut in as_completed(futures):
            t = futures[fut]
            ok = fut.result() is not None
            done += 1
            status = "ok " if ok else "FAIL"
            print(f"[{done}/{len(todo)}] {status} {t.stem}")

    write_manifest(targets, sheets_dir, out_dir / "manifest.csv")
    return targets


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base",
        type=Path,
        default=Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301"),
        help="Root of the CAMHD archive (contains YYYY/MM/DD/).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/home/jovyan/scaleworm-student-lab/scene_sorting"),
        help="Output directory for contact sheets and sort folders.",
    )
    p.add_argument("--start", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--weekday", type=int, default=0, help="Mon=0 ... Sun=6 (default 0=Monday)"
    )
    p.add_argument(
        "--step-weeks", type=int, default=1, help="Weeks between samples (default 1)"
    )
    p.add_argument(
        "--slots",
        nargs="+",
        default=list(SLOTS),
        help="HHMMSS slot strings to sample (default: all 8).",
    )
    p.add_argument("--every-s", type=float, default=30.0, help="Frame spacing (s).")
    p.add_argument("--cols", type=int, default=6, help="Contact-sheet columns.")
    p.add_argument("--workers", type=int, default=8, help=f"Max {MAX_WORKERS}.")
    p.add_argument(
        "--limit", type=int, default=None, help="Cap recordings (smoke test)."
    )
    p.add_argument("--dry-run", action="store_true", help="Enumerate only; no ffmpeg.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run(
        base=args.base,
        out_dir=args.out,
        start=args.start,
        end=args.end,
        weekday=args.weekday,
        step_weeks=args.step_weeks,
        slots=args.slots,
        every_s=args.every_s,
        cols=args.cols,
        workers=args.workers,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
