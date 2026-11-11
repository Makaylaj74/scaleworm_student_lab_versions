"""Build blind IRR round-4 packets: each counted frame comes with its time neighbours.

Round 4 follows Katie's recommendation: for every frame a rater annotates, they also
annotate the frame a few seconds **before** and **after** it, so a box that is really a
shadow (doesn't move with anything / changes with lighting) or a worm that moves can be
told apart. 5 anchor frames x 3 times (t-OFFSET, t, t+OFFSET) = **15 frames**, all
annotated blind from scratch, presented grouped by triplet. In the labeler the rater
can flip the view to the other two frames of the triplet with their current boxes
overlaid (clicking is disabled while viewing a neighbour).

Clean for EVERY rater incl. MJ: anchors are clear-era (2021-09..2023-08) Tue-Fri
recordings from the automated v8+SAHI harvest, on days where no recording has ever been
hand-counted, box-labelled or scene-sorted by anyone (any CAMHD stem under SEEN_SOURCES
blocks its whole day). Anchor time = the harvest's best_time_s. Density is stratified on
the harvest's detector count (one per calendar month, nearest the 10/30/50/70/90th
percentiles); that count is a selection aid only and is never written into a packet or
printed, so no rater sees a reference number. ``--pick`` walks outward from each
percentile and keeps the first candidate whose triplet passes QC; the result is frozen in
``ANCHORS`` for reproducibility.

Scene-1 needs a human eye: the harvest's auto-picked best_time_s often lands off Scene-1,
and image matching to a Monday Scene-1 frame can't tell (true Scene-1 a week apart scores
as low as off-scene views). So ``--scene-sheets`` writes N_PER_QUANTILE contact sheets per
quantile into SCENE_CHECK for a scene-only yes/no + time pass (MJ, 2026-11-10; no counts
shown), and ``--pick`` takes, per quantile, the first sheet sorted Scene-1 whose triplet at
the chosen time passes QC.

OFFSET_S = 10: a probe on 2022-11-07T12:15 showed the camera holds the identical Scene-1
framing for at least +/-20 s, and the Scene-1 timestamps are ~30 s precise, so +/-10 s
stays inside the hold. Triplet QC (both re-run on every build):
- registration: each neighbour's phase-correlation shift vs the anchor <= MAX_SHIFT_PX;
- matched focus: min/max of the triplet's sharpness (median 3x3-tile Laplacian variance)
  >= MIN_SHARP_RATIO. 0.5 was set on the first 5 candidate triplets by eye: the two that
  looked matched scored 0.77 / 0.55, the three with a visibly softer member 0.22-0.43.

Usage::

    python scripts/build_irr_triplet_packet.py --scene-sheets   # 1. candidate sheets
    #    2. scene-check them in notebooks/41_scene_check_irr_round4.ipynb
    python scripts/build_irr_triplet_packet.py --pick           # 3. print anchors
    python scripts/build_irr_triplet_packet.py MJ LG MS         # 4. build packets
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from datetime import date
from pathlib import Path

import build_irr_packet as bip
import matplotlib.image as mpimg
import numpy as np
from count_frames import extract_frame, stem_to_video
from scene_sampler import build_contact_sheet

ROUND = "round4"
OUT_DIR = bip.REPO / "validation/irr_round_4"
FRAME_CACHE = OUT_DIR / "frames"
OFFSET_S = 10
MAX_SHIFT_PX = 10.0
CLEAR_WINDOW = ("2021-09-01", "2023-08-31")
HARVEST = bip.REPO / "notebooks/harvest_clear_v8sahi_2021_2023.csv"
VIDEO_ROOT = Path("/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301")
# Everywhere a human has looked at / annotated recordings (labels, hand-counts, sorts,
# IRR packets + returns). A stem found here makes its whole day ineligible.
SEEN_SOURCES = [
    "validation",
    "datasets",
    "scene_sorting",
    "notebooks/handcount_2023_01.csv",
    "notebooks/handcount_2024_05.csv",
    "notebooks/LG_results",
    "notebooks/irr_packet_LG",
    "notebooks/MS_results_2",
]
# Round-4 working dirs: their stems are this round's own candidates, not prior exposure.
SCENE_CHECK = bip.REPO / "scene_sorting/irr_round4_scene_check"
EXEMPT = [OUT_DIR, SCENE_CHECK]
N_PER_QUANTILE = 4
# MJ's time first, then +/-10 s (still inside the 30 s contact-sheet tile she chose).
NUDGES_S = (0, -10, 10)
STEM_RE = re.compile(r"CAMHDA301-(20\d{6})T\d{6}")
QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
MIN_SHARP_RATIO = 0.5

# Output of --pick (2026-11-10): (frame_id, scene time s). Never-annotated days.
ANCHORS: list[tuple[str, int]] = []
EXTRA_FIELDS = ["triplet_id", "offset_s"]


def seen_days(sources: list[str] = SEEN_SOURCES, root: Path = bip.REPO) -> set[str]:
    """YYYYMMDD of every CAMHD recording any human has annotated/sorted in the repo."""
    days: set[str] = set()
    for src in sources:
        path = root / src
        files = (
            [path] if path.is_file() else (p for p in path.rglob("*") if p.is_file())
        )
        for f in files:
            if any(f.is_relative_to(e) for e in EXEMPT):
                continue
            days.update(STEM_RE.findall(f.name))
            if f.suffix in {".csv", ".txt", ".json", ".md"} and f.stat().st_size < 50e6:
                days.update(STEM_RE.findall(f.read_text(errors="ignore")))
    return days


def harvest_rows(path: Path = HARVEST) -> list[dict]:
    """Harvest recordings as manifest-shaped rows (+ private ``_density``)."""
    out = []
    for r in csv.DictReader(path.open(newline="")):
        fid = r["frame_id"]
        out.append(
            {
                "frame_id": fid,
                "datetime_utc": r["datetime_utc"][:19],
                "scene1_time_s": r["best_time_s"],
                "video_path": str(stem_to_video(fid, VIDEO_ROOT)),
                "_density": float(r["count_v8sahi_max"]),
            }
        )
    return out


def anchor_candidates(rows: list[dict], excluded_days: set[str]) -> list[list[dict]]:
    """Per density quantile, eligible never-seen Tue-Fri recordings, nearest-first."""
    lo, hi = CLEAR_WINDOW
    pool = []
    for r in rows:
        day = r["frame_id"].split("-")[1][:8]
        weekday = date(int(day[:4]), int(day[4:6]), int(day[6:])).weekday()
        if (
            1 <= weekday <= 4
            and lo <= r["datetime_utc"][:10] <= hi
            and day not in excluded_days
        ):
            pool.append(r)
    pool.sort(key=lambda r: (r["_density"], r["frame_id"]))
    out = []
    for q in QUANTILES:
        i = round(q * (len(pool) - 1))
        out.append(
            [pool[j] for j in sorted(range(len(pool)), key=lambda j: abs(j - i))]
        )
    return out


def neighbour_id(anchor: str, offset: int) -> str:
    """Frame id of the triplet member ``offset`` seconds from the anchor."""
    return f"{anchor}_dt{offset:+03d}"


def triplet_rows(
    anchor_rows: list[dict], fieldnames: list[str], offset_s: int = OFFSET_S
) -> list[dict]:
    """Blank manifest rows, ordered anchor by anchor as (t-offset, t, t+offset)."""
    out = []
    for k, a in enumerate(anchor_rows, start=1):
        for off in (-offset_s, 0, offset_s):
            row = bip.blank_manifest_rows([a], fieldnames)[0]
            fid = neighbour_id(a["frame_id"], off)
            row.update(
                frame_id=fid,
                label_path=f"labels/{fid}.txt",
                scene1_time_s=f"{float(a['scene1_time_s']) + off:g}",
                sharpness="",
                triplet_id=f"T{k}",
                offset_s=str(off),
            )
            out.append(row)
    return out


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(dy, dx) translation between two grayscale images by phase correlation."""
    fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    r = fa * np.conj(fb)
    corr = np.fft.ifft2(r / (np.abs(r) + 1e-12)).real
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = corr.shape
    return (dy - h if dy > h // 2 else dy), (dx - w if dx > w // 2 else dx)


def _gray(path: Path, step: int = 1) -> np.ndarray:
    return mpimg.imread(path)[::step, ::step, :3].mean(axis=-1)


def sharpness(gray: np.ndarray, n: int = 3) -> float:
    """Median over an n x n tile grid of the Laplacian variance (robust to one bad patch)."""
    g = gray * 255
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    h, w = lap.shape
    return float(
        np.median(
            [
                lap[i * h // n : (i + 1) * h // n, j * w // n : (j + 1) * w // n].var()
                for i in range(n)
                for j in range(n)
            ]
        )
    )


def cached(row: dict, cache: Path = FRAME_CACHE) -> Path:
    """Cache path keyed on frame id AND video time (a stem can be tried at several times)."""
    return cache / f"{row['frame_id']}_at{float(row['scene1_time_s']):g}s.png"


def extract(rows: list[dict], cache: Path = FRAME_CACHE) -> None:
    """Extract every listed frame into the cache (skips ones already there)."""
    cache.mkdir(parents=True, exist_ok=True)
    for r in rows:
        out = cached(r, cache)
        if not out.exists() and not extract_frame(
            Path(r["video_path"]), float(r["scene1_time_s"]), out
        ):
            raise RuntimeError(f"ffmpeg failed for {r['frame_id']}")


def check_triplet(paths: list[Path], step: int = 4) -> tuple[bool, str]:
    """QC one (t-offset, t, t+offset) triplet: registration + matched focus."""
    grays = [_gray(p) for p in paths]
    shifts = [
        step
        * float(np.hypot(*phase_shift(grays[1][::step, ::step], g[::step, ::step])))
        for g in (grays[0], grays[2])
    ]
    sharp = [sharpness(g) for g in grays]
    ratio = min(sharp) / max(sharp)
    ok = max(shifts) <= MAX_SHIFT_PX and ratio >= MIN_SHARP_RATIO
    msg = (
        f"shift {shifts[0]:.0f}/{shifts[1]:.0f} px, sharpness "
        f"{'/'.join(f'{v:.0f}' for v in sharp)} (ratio {ratio:.2f})"
    )
    return ok, msg


def qc_rows(rows: list[dict], cache: Path = FRAME_CACHE) -> list[tuple[str, bool, str]]:
    """Extract + QC every triplet in ``rows`` (as made by triplet_rows)."""
    extract(rows, cache)
    out = []
    for k in range(0, len(rows), 3):
        trip = rows[k : k + 3]
        ok, msg = check_triplet([cached(r, cache) for r in trip])
        out.append((trip[1]["frame_id"].rsplit("_dt", 1)[0], ok, msg))
    return out


def scene_candidates(rows: list[dict], excluded_days: set[str]) -> list[dict]:
    """N_PER_QUANTILE recordings per quantile, nearest-first, distinct months overall."""
    months: set[str] = set()
    out = []
    for q, cands in enumerate(anchor_candidates(rows, excluded_days)):
        taken = 0
        for c in cands:
            month = c["frame_id"][10:16]
            if month in months:
                continue
            months.add(month)
            out.append({**c, "quantile": q})
            taken += 1
            if taken == N_PER_QUANTILE:
                break
    return out


def more_candidates(
    rows: list[dict],
    excluded_days: set[str],
    q: int,
    n: int,
    session: Path = SCENE_CHECK,
) -> list[dict]:
    """n further candidates for quantile q: new months first, then any unused day."""
    have = list(csv.DictReader((session / "candidates.csv").open()))
    months = {c["stem"][10:16] for c in have}
    days = {c["stem"][10:18] for c in have}
    ranked = anchor_candidates(rows, excluded_days)[q]
    ranked = sorted(ranked, key=lambda c: c["frame_id"][10:16] in months)  # stable
    out = []
    for c in ranked:
        if c["frame_id"][10:18] not in days:
            days.add(c["frame_id"][10:18])
            out.append({**c, "quantile": q})
            if len(out) == n:
                break
    return out


def build_scene_sheets(
    cands: list[dict], session: Path = SCENE_CHECK, append: bool = False
) -> None:
    """Contact sheets + manifest in the scene_sorter layout (no density written)."""
    sheets = session / "contact_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with (session / "manifest.csv").open(mode, newline="") as fh:
        w = csv.writer(fh)
        if not append:
            w.writerow(
                ["stem", "datetime_utc", "recording_date", "sheet_path", "video_path"]
            )
        for c in cands:
            w.writerow(
                [
                    c["frame_id"],
                    c["datetime_utc"],
                    c["datetime_utc"][:10],
                    f"contact_sheets/{c['frame_id']}.png",
                    c["video_path"],
                ]
            )
    with (session / "candidates.csv").open(mode, newline="") as fh:
        w = csv.writer(fh)
        if not append:
            w.writerow(["stem", "quantile"])
        w.writerows((c["frame_id"], c["quantile"]) for c in cands)
    for c in cands:
        out = sheets / f"{c['frame_id']}.png"
        if not out.exists():
            build_contact_sheet(Path(c["video_path"]), out)
        print(f"  sheet {c['frame_id']}")


def first_passing(stem: str, t: float, fieldnames: list[str]) -> dict | None:
    """The chosen time, else a NUDGES_S shift of it (stays inside the 30 s sheet tile)."""
    for nudge in NUDGES_S:
        a = {
            "frame_id": stem,
            "scene1_time_s": f"{t + nudge:g}",
            "video_path": str(stem_to_video(stem, VIDEO_ROOT)),
        }
        [(fid, ok, msg)] = qc_rows(triplet_rows([a], fieldnames))
        print(f"    {'PASS' if ok else 'fail'} {fid} @{t + nudge:g}s: {msg}")
        if ok:
            return a
    return None


def pick_anchors(fieldnames: list[str], session: Path = SCENE_CHECK) -> list[dict]:
    """Per quantile, the first Scene-1-sorted candidate whose triplet passes QC."""
    log = {r["stem"]: r for r in csv.DictReader((session / "sort_log.csv").open())}
    cands = list(csv.DictReader((session / "candidates.csv").open()))
    picked = []
    for q in sorted({c["quantile"] for c in cands}, key=int):
        print(f"  q{q}")
        for c in (c for c in cands if c["quantile"] == q):
            d = log.get(c["stem"])
            if d is None or d["decision"] != "scene1":
                continue
            a = first_passing(c["stem"], float(d["scene1_time_s"]), fieldnames)
            if a:
                picked.append(a)
                break
        else:
            print(f"  q{q}: NO passing Scene-1 candidate")
    return picked


def triplet_view_source() -> str:
    return '''class TripletLabeler(BlindLabeler):
    """BlindLabeler + flip the view to the other two frames of the same triplet."""

    def __init__(self, stems):
        self.view = widgets.ToggleButtons(options=["-", "-", "-"],
                                          description="View time:")
        self.hint = widgets.HTML()
        self._loading = False
        super().__init__(stems)
        self.view.observe(lambda ch: self._show_view(), names="value")
        self.box.children = (self.box.children[0], self.box.children[1],
                             widgets.HBox([self.view, self.hint]),
                             *self.box.children[2:])

    def _triplet(self):
        tid = INFO[self._stem()]["triplet_id"]
        members = sorted((int(INFO[s]["offset_s"]), s) for s in self.stems
                         if INFO[s]["triplet_id"] == tid)
        return {f"{off:+d} s": s for off, s in members}

    def _own_label(self):
        return f"{int(INFO[self._stem()]['offset_s']):+d} s"

    def _load(self):
        super()._load()
        self._viewing = self._own_label()
        self._loading = True
        self.view.options = list(self._triplet())
        self.view.value = self._viewing
        self._loading = False
        self.hint.value = ""

    def _show_view(self):
        label = self.view.value
        if self._loading or label is None or label == self._viewing:
            return
        self._viewing = label
        stem = self._triplet()[label]
        self.ax.images[0].set_data(mpimg.imread(IMAGES / f"{stem}.png"))
        own = label == self._own_label()
        self.hint.value = "" if own else (
            f"<b style='color:#D55E00'>Viewing the {label} frame with THIS frame's "
            f"boxes on top -- clicking is off. Switch back to {self._own_label()} "
            "to keep drawing.</b>")
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if self._viewing != self._own_label():
            return
        super()._on_click(event)


app = TripletLabeler(stems)
display(app.box)'''


def build_notebook(rater: str, packet: str, n_frames: int) -> dict:
    """Blind labeler notebook with the triplet view toggle."""
    nb = bip.build_notebook(rater, n_frames)

    def code(src):
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src.splitlines(keepends=True),
        }

    base = bip._labeler_source().rsplit("\n\n\napp = ", 1)[0]
    nb["cells"][2] = code(bip._config_source(rater, packet))
    nb["cells"][3] = code(base + "\n\n\n" + triplet_view_source())
    md = "".join(nb["cells"][0]["source"]).replace(
        "Resumable:",
        "Frames come in **triplets** (the same view ~10 s before, at, and after a "
        "moment). Annotate all three. Use **View time** to flip to the other two "
        "frames with your boxes on top, to check whether a boxed thing is a worm "
        "that moved or a shadow.\n\nResumable:",
    )
    nb["cells"][0]["source"] = md.splitlines(keepends=True)
    return nb


def readme_source(rater: str, packet: str, n_frames: int) -> str:
    text = bip._readme_source(rater, n_frames).replace(f"irr_packet_{rater}", packet)
    triplet = f"""### This round: pictures come in threes
The {n_frames} pictures are **{n_frames // 3} groups of 3**: the same camera view about
**{OFFSET_S} seconds before**, **at**, and **{OFFSET_S} seconds after** a moment. They
come one after another in the labeler. **Box the worms in all three, each on its own.**

Above each picture there is a **View time** switch (`-{OFFSET_S} s` / `+0 s` /
`+{OFFSET_S} s`). Flip it to see the other two pictures of the group **with the boxes
you've drawn so far laid on top**. Use it to check things you're unsure about:
- a dark shape that **stays exactly put while the lighting/edges change** may be a
  **shadow**, not a worm;
- something that **shifts or bends** between the pictures is likely a **living worm**.

You can only add boxes while the switch is on the picture you're working on (it jumps
back automatically when you go to the next picture).

"""
    return text.replace("### What counts as", triplet + "### What counts as", 1)


def build_packet(
    rater: str,
    rows: list[dict],
    fieldnames: list[str],
    out_dir: Path,
    cache: Path = FRAME_CACHE,
) -> Path:
    packet = f"irr_packet_{rater}_{ROUND}"
    pkt = out_dir / packet
    if pkt.exists():
        shutil.rmtree(pkt)
    (pkt / "frames").mkdir(parents=True)
    (pkt / "labels").mkdir()
    for r in rows:
        shutil.copy2(cached(r, cache), pkt / "frames" / f"{r['frame_id']}.png")
    with (pkt / "irr_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    (pkt / "labeler.ipynb").write_text(
        json.dumps(build_notebook(rater, packet, len(rows)), indent=1)
    )
    (pkt / "README.md").write_text(readme_source(rater, packet, len(rows)))
    (pkt / "labels" / ".gitkeep").write_text("")

    zip_path = out_dir / f"{packet}.zip"
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(pkt.rglob("*")):
            zf.write(p, p.relative_to(out_dir))
    print(
        f"  {rater}: {len(rows)} frames -> {zip_path.name} "
        f"({zip_path.stat().st_size / 1e6:.1f} MB)"
    )
    return zip_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raters", nargs="*", help="rater ids / initials, e.g. LG MS")
    ap.add_argument(
        "--scene-sheets",
        action="store_true",
        help="write candidate contact sheets for the scene check",
    )
    ap.add_argument(
        "--more",
        nargs=2,
        type=int,
        metavar=("Q", "N"),
        help="append N more scene-check sheets for quantile index Q",
    )
    ap.add_argument("--pick", action="store_true", help="print the anchor selection")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "packets")
    args = ap.parse_args(argv)

    fieldnames = list(csv.DictReader(bip.MANIFEST.open(newline="")).fieldnames)
    fieldnames += EXTRA_FIELDS
    excluded = seen_days()
    if args.scene_sheets:
        build_scene_sheets(scene_candidates(harvest_rows(), excluded))
        return
    if args.more:
        q, n = args.more
        build_scene_sheets(more_candidates(harvest_rows(), excluded, q, n), append=True)
        return
    if args.pick:
        for r in pick_anchors(fieldnames):
            print(f'    ("{r["frame_id"]}", {r["scene1_time_s"]}),')
        return
    if not ANCHORS:
        raise SystemExit(
            "ANCHORS is empty: run --scene-sheets, scene-check, --pick first"
        )
    leaked = [a for a, _ in ANCHORS if a.split("-")[1][:8] in excluded]
    if leaked:
        raise RuntimeError(f"anchors on already-annotated days: {leaked}")
    anchors = [
        {
            "frame_id": a,
            "datetime_utc": f"{a[10:14]}-{a[14:16]}-{a[16:18]}T{a[19:21]}:{a[21:23]}:{a[23:25]}",
            "scene1_time_s": str(t),
            "video_path": str(stem_to_video(a, VIDEO_ROOT)),
        }
        for a, t in ANCHORS
    ]
    rows = triplet_rows(anchors, fieldnames)
    print(
        f"IRR {ROUND}: {len(ANCHORS)} triplets x 3 = {len(rows)} frames "
        f"(offset +/-{OFFSET_S} s)"
    )
    for fid, ok, msg in qc_rows(rows):
        print(f"  {'PASS' if ok else 'FAIL'} {fid}: {msg}")
        if not ok:
            raise RuntimeError(f"triplet {fid} failed QC")
    args.out.mkdir(parents=True, exist_ok=True)
    for rater in args.raters:
        build_packet(rater, rows, fieldnames, args.out)


if __name__ == "__main__":
    main()
