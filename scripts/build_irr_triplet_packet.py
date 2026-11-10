"""Build blind IRR round-4 packets: each counted frame comes with its time neighbours.

Round 4 follows Katie's recommendation: for every frame a rater annotates, they also
annotate the frame a few seconds **before** and **after** it, so a box that is really a
shadow (doesn't move with anything / changes with lighting) or a worm that moves can be
told apart. 5 anchor frames x 3 times (t-OFFSET, t, t+OFFSET) = **15 frames**, all
annotated blind from scratch, presented grouped by triplet. In the labeler the rater
can flip the view to the other two frames of the triplet with their current boxes
overlaid (clicking is disabled while viewing a neighbour).

Anchors are clear-era (2021-09..2023-08) counted Monday frames on days never used in
IRR rounds 2/3, one per calendar month, taken nearest the 10/30/50/70/90th percentiles of
MJ's Monday count so the set spans low to high density. ``--pick`` walks outward from each
percentile and keeps the first candidate whose triplet passes QC; the result is frozen in
``ANCHORS`` for reproducibility.

OFFSET_S = 10: a probe on 2022-11-07T12:15 showed the camera holds the identical Scene-1
framing for at least +/-20 s, and the Scene-1 timestamps are ~30 s precise, so +/-10 s
stays inside the hold. Triplet QC (both re-run on every build):
- registration: each neighbour's phase-correlation shift vs the anchor <= MAX_SHIFT_PX;
- matched focus: min/max of the triplet's sharpness (median 3x3-tile Laplacian variance)
  >= MIN_SHARP_RATIO. 0.5 was set on the first 5 candidate triplets by eye: the two that
  looked matched scored 0.77 / 0.55, the three with a visibly softer member 0.22-0.43.

Usage::

    python scripts/build_irr_triplet_packet.py LG MS
    python scripts/build_irr_triplet_packet.py --pick      # print the anchor selection
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import build_irr_packet as bip
import matplotlib.image as mpimg
import numpy as np
from count_frames import extract_frame

ROUND = "round4"
OUT_DIR = bip.REPO / "validation/irr_round_4"
FRAME_CACHE = OUT_DIR / "frames"
OFFSET_S = 10
MAX_SHIFT_PX = 10.0
CLEAR_WINDOW = ("2021-09-01", "2023-08-31")
USED_DAYS = {
    d for periods in bip.ROUNDS.values() for days in periods.values() for d in days
}
QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
MIN_SHARP_RATIO = 0.5

# Output of --pick (2026-11-10); MJ Monday counts 7 / 10 / 13 / 17 / 25.
ANCHORS = [
    "CAMHDA301-20230626T031500",
    "CAMHDA301-20230327T091500",
    "CAMHDA301-20230821T151500",
    "CAMHDA301-20220919T211500",
    "CAMHDA301-20220829T211500",
]
EXTRA_FIELDS = ["triplet_id", "offset_s"]


def anchor_candidates(rows: list[dict]) -> list[list[dict]]:
    """Per count quantile, eligible counted clear-era frames ordered nearest-first."""
    lo, hi = CLEAR_WINDOW
    pool = [
        r
        for r in rows
        if r["frame_status"] == "counted"
        and lo <= r["datetime_utc"][:10] <= hi
        and r["frame_id"].split("-")[1][:8] not in USED_DAYS
    ]
    pool.sort(key=lambda r: (float(r["worm_count"]), r["frame_id"]))
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


def extract(rows: list[dict], cache: Path = FRAME_CACHE) -> None:
    """Extract every listed frame into the cache (skips ones already there)."""
    cache.mkdir(parents=True, exist_ok=True)
    for r in rows:
        out = cache / f"{r['frame_id']}.png"
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
        ok, msg = check_triplet([cache / f"{r['frame_id']}.png" for r in trip])
        out.append((trip[1]["frame_id"].rsplit("_dt", 1)[0], ok, msg))
    return out


def pick_anchors(
    rows: list[dict], fieldnames: list[str], max_tries: int = 8
) -> list[dict]:
    """Nearest-to-quantile anchors, one per month, whose triplet passes QC."""
    picked: list[dict] = []
    months: set[str] = set()
    for cands in anchor_candidates(rows):
        tries = 0
        for c in cands:
            month = c["frame_id"].split("-")[1][:6]
            if month in months:
                continue
            tries += 1
            [(fid, ok, msg)] = qc_rows(triplet_rows([c], fieldnames))
            print(f"  {'PASS' if ok else 'fail'} {fid} (MJ {c['worm_count']}): {msg}")
            if ok:
                picked.append(c)
                months.add(month)
                break
            if tries >= max_tries:
                raise RuntimeError("no passing triplet near this quantile")
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
        shutil.copy2(cache / f"{r['frame_id']}.png", pkt / "frames")
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
    ap.add_argument("--pick", action="store_true", help="print the anchor selection")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "packets")
    args = ap.parse_args(argv)

    manifest = list(csv.DictReader(bip.MANIFEST.open(newline="")))
    fieldnames = list(manifest[0]) + EXTRA_FIELDS
    if args.pick:
        for r in pick_anchors(manifest, fieldnames):
            print(f'    "{r["frame_id"]}",  # MJ count {r["worm_count"]}')
        return
    by_id = {r["frame_id"]: r for r in manifest}
    rows = triplet_rows([by_id[a] for a in ANCHORS], fieldnames)
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
