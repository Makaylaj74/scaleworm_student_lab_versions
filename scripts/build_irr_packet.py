"""Build a self-contained, *blind* IRR annotation packet for a lab partner.

Round-2 IRR compares independent manual worm **counts/boxes** on a shared frame set
(low-abundance Jan-2022 vs peak Jan-2023 Mondays). Each partner gets a zip they unzip
into their home directory; they draw every worm from scratch (no pre-labels -- the
`from_scratch` method in the schema, which is what makes this measure *independent*
reliability rather than shared-anchor correction), then send back only their tiny
`labels/` + `irr_manifest.csv` (a few hundred KB of text -- the ~5 MB PNGs never come
back). See `validation/irr_round_2/README.md` for the round-trip.

Packet layout (self-contained -- nothing in it points back into the repo)::

    irr_packet_<RATER>/
        README.md            # rater-facing instructions
        labeler.ipynb        # blind box labeler, ROOT + COUNTER pre-filled
        frames/<id>.png      # the IRR images
        labels/              # empty; the labeler writes <id>.txt here
        irr_manifest.csv     # one blank row per frame (frame_status=pending)

Usage::

    python scripts/build_irr_packet.py LG MS          # one packet each
    python scripts/build_irr_packet.py MJ --slots 4   # optional blind self-pass
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

REPO = Path("/home/jovyan/scaleworm-student-lab")
SERIES = REPO / "validation/monday_manual_series"
MANIFEST = SERIES / "frame_manifest.csv"
IMAGES = SERIES / "images"
OUT_DIR = REPO / "validation/irr_round_2/packets"

# IRR frame selection: the low-abundance and peak Mondays chosen with Makayla.
# Only Mondays that are actually sorted+counted in the series are listed here.
PERIODS = {
    "low_jan2022": ["20220117", "20220131"],
    "peak_jan2023": ["20230109", "20230116", "20230123"],
}
SLOTS_PER_DAY = 4  # cap per day, evenly spaced across the available 3-hourly slots


def pick_even(items: list[str], k: int) -> list[str]:
    """Pick up to k items evenly spaced across a sorted list (endpoints included)."""
    if k >= len(items) or k <= 0:
        return list(items)
    if k == 1:
        return [items[0]]
    idx = {round(i * (len(items) - 1) / (k - 1)) for i in range(k)}
    return [items[i] for i in sorted(idx)]


def select_frames(slots_per_day: int = SLOTS_PER_DAY) -> list[dict]:
    """Return the manifest rows for the IRR frame set (counted frames only)."""
    rows = list(csv.DictReader(MANIFEST.open(newline="")))
    by_id = {r["frame_id"]: r for r in rows}
    wanted_days = {d for days in PERIODS.values() for d in days}
    # group counted frames by their YYYYMMDD day
    per_day: dict[str, list[str]] = {}
    for fid, r in by_id.items():
        day = fid.split("-")[1][:8]
        if day in wanted_days and r["frame_status"] == "counted":
            per_day.setdefault(day, []).append(fid)
    selected: list[dict] = []
    for day in sorted(wanted_days):
        frames = sorted(per_day.get(day, []))
        if not frames:
            print(f"  WARNING: no counted frames for day {day} -- skipped")
            continue
        chosen = pick_even(frames, slots_per_day)
        selected.extend(by_id[f] for f in chosen)
    return selected


def blank_manifest_rows(rows: list[dict], fieldnames: list[str]) -> list[dict]:
    """Copy provenance columns; blank out everything an annotator must fill."""
    keep = {"frame_id", "datetime_utc", "camera_unit", "quarter",
            "scene1_time_s", "video_path", "sharpness", "split"}
    out = []
    for r in rows:
        blank = {k: "" for k in fieldnames}
        for k in keep:
            if k in r:
                blank[k] = r[k]
        blank["frame_status"] = "pending"
        blank["label_path"] = f"labels/{r['frame_id']}.txt"
        blank["prelabel_model"] = "none"  # blind: no pre-labels shipped
        out.append(blank)
    return out


def _md_source(rater: str, n_frames: int) -> str:
    return (
        "*AI-generated draft (Claude, Anthropic) -- for review. The labeling UI is "
        "version-controlled; the boxes you draw are your own independent ground "
        "truth.*\n\n"
        "<span style=\"font-family: 'Courier New', monospace;\">\n\n"
        f"# IRR blind box labeler -- rater `{rater}`\n\n"
        f"You have **{n_frames} frames**. For each frame, **draw a box on every "
        "scaleworm you can see** -- this is a *blind* count, so there are no "
        "pre-drawn boxes; you start from an empty frame.\n\n"
        "**Kernel:** use your normal **Python 3** kernel -- no special or shared "
        "kernel is needed. The first cell installs the clicking backend (`ipympl`) "
        "into that kernel automatically the first time you run it. If no clickable "
        "picture appears, reload the browser tab and run **Kernel > Restart & Run "
        "All** once more.\n\n"
        "**How to use**\n"
        "1. **Left-click each worm** -> drops a box on it.\n"
        "2. **Right-click** a box to delete it; **Undo** / **Clear** for last / all.\n"
        "3. **Save & Next ▶** -> writes your boxes and sets the row to `counted` "
        "(`worm_count` = number of boxes -- never typed by hand).\n"
        "4. **Unusable (blur)** / **No Scene-1** -> the frame is *missing data*, not "
        "a zero (`worm_count` = NA); use these only if you genuinely cannot count.\n\n"
        "Resumable: reopens on the first frame you have not yet saved. When you reach "
        "the end, follow the README to send `labels/` + `irr_manifest.csv` back.\n\n"
        "</span>"
    )


def _setup_source() -> str:
    return '''# --- One-time setup: make THIS kernel able to do the click-to-label ---
# You do NOT need any special or shared kernel -- your normal Python 3 kernel
# is fine. This installs the interactive plotting backend (ipympl) into it the
# first time only. Safe to re-run: it does nothing if ipympl is already present.
try:
    import ipympl  # noqa: F401
except ModuleNotFoundError:
    import subprocess
    import sys

    print("Installing the clicking backend (ipympl) -- about 30 seconds...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ipympl"], check=True)
    print(
        "Done. If no clickable picture appears below, reload this browser tab,\\n"
        "then run  Kernel > Restart Kernel and Run All Cells  once more."
    )'''


def _config_source(rater: str) -> str:
    return f'''%matplotlib widget
import csv
import statistics
from datetime import date
from pathlib import Path

import ipywidgets as widgets
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from IPython.display import display

# If you unzipped the packet somewhere other than your home directory,
# edit this ONE line to point at the unzipped irr_packet_{rater} folder.
ROOT = Path.home() / "irr_packet_{rater}"

IMAGES = ROOT / "frames"
LABELS = ROOT / "labels"
MANIFEST = ROOT / "irr_manifest.csv"
LABELS.mkdir(exist_ok=True)

COUNTER = "{rater}"            # your initials -> manifest `counter`
TODAY = date.today().isoformat()
DEFAULT_W, DEFAULT_H = 0.0342, 0.0585  # median normalized worm box (fallback size)
DONE = {{"counted", "unusable_blur", "no_scene1"}}


def read_manifest():
    with MANIFEST.open(newline="") as fh:
        r = csv.DictReader(fh)
        return list(r), r.fieldnames


def write_manifest(rows, fields):
    with MANIFEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def update_row(frame_id, **updates):
    rows, fields = read_manifest()
    for row in rows:
        if row["frame_id"] == frame_id:
            row.update(updates)
            break
    write_manifest(rows, fields)


def load_yolo(path, W, H):
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                cx, cy, w, h = (float(v) for v in p[1:])
                cx, cy, w, h = cx * W, cy * H, w * W, h * H
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return boxes


def save_yolo(path, boxes, W, H):
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        w, h = abs(x2 - x1) / W, abs(y2 - y1) / H
        lines.append(f"0 {{cx:.6f}} {{cy:.6f}} {{w:.6f}} {{h:.6f}}")
    path.write_text("\\n".join(lines) + ("\\n" if lines else ""))


ROWS, _FIELDS = read_manifest()
INFO = {{r["frame_id"]: r for r in ROWS}}
stems = [r["frame_id"] for r in ROWS]
n_done = sum(1 for r in ROWS if r["frame_status"] in DONE)
print(f"{{len(stems)}} frames  ({{n_done}} done, {{len(stems) - n_done}} remaining).")
print("Left-click each worm; right-click a box to delete; then Save & Next.")'''


def _labeler_source() -> str:
    return '''class BlindLabeler:
    """Draw a box on every worm from scratch; worm_count is derived from the boxes."""

    def __init__(self, stems):
        self.stems = stems
        self.pos = next(
            (i for i, s in enumerate(stems) if INFO[s]["frame_status"] not in DONE), 0
        )
        self.boxes = []
        self.coll = None
        self.bw, self.bh = DEFAULT_W, DEFAULT_H

        def mk(desc, style=""):
            return widgets.Button(description=desc, button_style=style,
                                  layout=widgets.Layout(width="auto"))

        self.b_undo = mk("Undo")
        self.b_clear = mk("Clear")
        self.b_prev = mk("◀ Prev")
        self.b_save = mk("Save & Next ▶", "success")
        self.b_blur = mk("Unusable (blur)", "danger")
        self.b_nos1 = mk("No Scene-1", "warning")
        self.b_skip = mk("Skip")
        self.b_undo.on_click(lambda _: self._undo())
        self.b_clear.on_click(lambda _: self._clear())
        self.b_prev.on_click(lambda _: self._prev())
        self.b_save.on_click(lambda _: self._save())
        self.b_blur.on_click(lambda _: self._mark("unusable_blur"))
        self.b_nos1.on_click(lambda _: self._mark("no_scene1"))
        self.b_skip.on_click(lambda _: self._advance())
        self.status = widgets.HTML()
        self.msg = widgets.Output()

        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        plt.ion()
        self.fig.canvas.header_visible = False
        self.fig.canvas.toolbar_position = "right"
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        controls = widgets.HBox([self.b_undo, self.b_clear, self.b_prev,
                                 self.b_save, self.b_blur, self.b_nos1, self.b_skip])
        self.box = widgets.VBox([controls, self.status, self.fig.canvas, self.msg])
        self._load()

    def _stem(self):
        return self.stems[self.pos]

    def _load(self):
        stem = self._stem()
        img = mpimg.imread(IMAGES / f"{stem}.png")
        self.H, self.W = img.shape[:2]
        self.ax.clear()
        self.coll = None
        self.ax.imshow(img)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.boxes = load_yolo(LABELS / f"{stem}.txt", self.W, self.H)
        if len(self.boxes) >= 3:
            self.bw = statistics.median((x2 - x1) / self.W for x1, _, x2, _ in self.boxes)
            self.bh = statistics.median((y2 - y1) / self.H for _, y1, _, y2 in self.boxes)
        else:
            self.bw, self.bh = DEFAULT_W, DEFAULT_H
        self._draw()
        self._status()

    def _draw(self):
        if self.coll is not None:
            try:
                self.coll.remove()
            except ValueError:
                pass
        rects = [mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1)
                 for x1, y1, x2, y2 in self.boxes]
        self.coll = PatchCollection(rects, facecolor="none",
                                    edgecolor="#D55E00", linewidths=1.6)
        self.ax.add_collection(self.coll)
        info = INFO[self._stem()]
        self.ax.set_title(
            f"{self._stem()}  [{info['split']}]   —   draw ALL worms   —   "
            f"{len(self.boxes)} boxes", fontsize=10)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if getattr(self.fig.canvas, "toolbar", None) and self.fig.canvas.toolbar.mode != "":
            return  # zoom/pan active
        if event.button == 1:
            w, h = self.bw * self.W, self.bh * self.H
            cx, cy = event.xdata, event.ydata
            self.boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            self._draw()
        elif event.button == 3:
            hit = [i for i, (x1, y1, x2, y2) in enumerate(self.boxes)
                   if x1 <= event.xdata <= x2 and y1 <= event.ydata <= y2]
            if hit:
                self.boxes.pop(hit[-1])
                self._draw()

    def _undo(self):
        if self.boxes:
            self.boxes.pop()
            self._draw()

    def _clear(self):
        self.boxes = []
        self._draw()

    def _status(self):
        n_done = sum(1 for s in self.stems if INFO[s]["frame_status"] in DONE)
        st = INFO[self._stem()]["frame_status"]
        self.status.value = (
            f"<b>Frame {self.pos + 1}/{len(self.stems)}</b> &nbsp; {self._stem()} "
            f"&nbsp;<i>(status: {st})</i> &nbsp;&mdash;&nbsp; "
            f"<b>{n_done}</b> done, <b>{len(self.stems) - n_done}</b> left")

    def _save(self):
        stem = self._stem()
        save_yolo(LABELS / f"{stem}.txt", self.boxes, self.W, self.H)
        update_row(stem, frame_status="counted", worm_count=str(len(self.boxes)),
                   label_path=f"labels/{stem}.txt", annotation_method="from_scratch",
                   counter=COUNTER, date_counted=TODAY)
        INFO[stem]["frame_status"] = "counted"
        INFO[stem]["worm_count"] = str(len(self.boxes))
        self._advance()

    def _mark(self, status):
        stem = self._stem()
        update_row(stem, frame_status=status, worm_count="",
                   annotation_method="from_scratch",
                   counter=COUNTER, date_counted=TODAY)
        INFO[stem]["frame_status"] = status
        INFO[stem]["worm_count"] = ""
        self._advance()

    def _advance(self):
        self.msg.clear_output()
        if self.pos < len(self.stems) - 1:
            self.pos += 1
            self._load()
        else:
            with self.msg:
                print("✅ Last frame saved. Now send labels/ + irr_manifest.csv "
                      "back (see README).")

    def _prev(self):
        if self.pos > 0:
            self.pos -= 1
            self._load()


app = BlindLabeler(stems)
display(app.box)'''


def build_notebook(rater: str, n_frames: int) -> dict:
    """Assemble the blind labeler notebook (nbformat 4) for one rater."""
    def code(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": src.splitlines(keepends=True)}

    def md(src):
        return {"cell_type": "markdown", "metadata": {},
                "source": src.splitlines(keepends=True)}

    return {
        "cells": [
            md(_md_source(rater, n_frames)),
            code(_setup_source()),
            code(_config_source(rater)),
            code(_labeler_source()),
        ],
        "metadata": {
            # Generic default kernel present on every Jupyter install -- the packet
            # is self-contained (the setup cell installs ipympl), so raters never
            # need a special or shared kernel.
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _readme_source(rater: str, n_frames: int) -> str:
    return f"""*AI-generated draft (Claude, Anthropic) -- for review. Steps derive from the
version-controlled `build_irr_packet.py` / `irr_counts_agreement.py`. The "what counts as
one worm" rules below are a draft for Makayla to confirm before this goes out.*

# How to count the scaleworms -- rater {rater}

Hi! Thanks for helping check the scaleworm counts. Your job is simple: look at each
picture and **put one box on every worm you can see.** That's it -- you don't type any
numbers, and you don't have to sort or choose the pictures. The **{n_frames} pictures**
in this packet are already chosen and ready.

You'll do this on your own, without seeing anyone else's answers -- that's the whole
point. We then check how closely your counts line up with the other counters. Plan for
about **1 hour**. You can stop and come back anytime; it remembers where you left off.

---

## Part 1 -- Open it (do this once)

1. **Unzip** `irr_packet_{rater}.zip` into your **home folder**. Afterward you should
   have a folder called `irr_packet_{rater}` sitting in your home directory
   (the file browser on the left, top level).
2. Inside that folder, **double-click `labeler.ipynb`** to open it.
3. At the **top-right of the notebook**, the kernel should say **Python 3** (the
   default). If a small window pops up asking you to *Select Kernel*, just pick
   **Python 3** and click Select. You do **not** need any special or shared kernel.
4. In the top menu, click **Kernel ▸ Restart Kernel and Run All Cells…** and confirm.
5. **The first time only,** the top cell spends about 30 seconds installing a small
   plotting add-on. When a **picture with buttons above it** appears, you're ready. If
   the area under the cells instead looks empty, **reload this browser tab** and do
   step 4 once more — it will be instant the second time.

---

## Part 2 -- Count (the actual work)

For each picture:

- **Left-click once on each worm.** A small orange box drops where you clicked. One
  click = one worm.
- Made a mistake? **Right-click on a box** to delete just that box. Or use **Undo**
  (removes your last box) / **Clear** (removes all boxes on this picture).
- When you've boxed every worm in the picture, click the green **Save & Next ▶** button.
  It saves and moves to the next picture. The count is just how many boxes you drew --
  you never type a number.
- **◀ Prev** goes back if you want to re-check a previous picture.

**Two special buttons -- use them rarely:**
- **Unusable (blur)** -- the picture is too blurry/dark to count at all.
- **No Scene-1** -- the camera isn't actually showing the vent (wrong view).

Only use those if you genuinely *can't* count the frame. They mark it "skip this one"
(which is different from "zero worms"). For this packet almost every picture should be
countable, so you'll mostly just be clicking worms and hitting **Save & Next**.

### What counts as "one worm"  *(draft -- Makayla to confirm)*
- Box each **individual worm body you can make out**, one box per worm.
- If worms overlap or clump, count as many **distinct** ones as you can separate by eye;
  don't guess at ones you can't actually distinguish.
- A worm **half-cut-off at the edge** still counts if you can tell it's a worm.
- Count the **worms**, not the white tubes/background texture.
- Don't count the same worm twice. When unsure whether a faint shape is a worm, use your
  best judgment -- there are no trick questions, and it's fine if you're occasionally
  unsure. Just be consistent with yourself across all the pictures.

### Tips
- Use the **zoom/pan tools** on the right edge of the picture to look closely, then click
  the tool off again before clicking worms (while zoom is active, clicks zoom instead of
  adding boxes).
- Take breaks. It's **resumable** -- next time you Restart & Run All, it jumps to the
  first picture you haven't saved yet. Keep going until you've saved all {n_frames}.

---

## Part 3 -- Send your answers back

When you've saved all {n_frames} pictures, make a small results file. Open a Terminal
(File ▸ New ▸ Terminal) and paste:

```bash
cd ~/irr_packet_{rater}
zip -r {rater}_results.zip labels/ irr_manifest.csv
```

That makes `{rater}_results.zip` (only a few hundred KB). **Email that one file to
Makayla** (or drop it in the shared Drive folder).

⚠️ **Do not send the pictures back** -- they're large and Makayla already has them. Only
the little `{rater}_results.zip` (the `labels/` folder + `irr_manifest.csv`) needs to
come back.

That's everything -- thank you so much!
"""


def build_packet(rater: str, slots_per_day: int, out_dir: Path) -> Path:
    rows = select_frames(slots_per_day)
    fieldnames = list(csv.DictReader(MANIFEST.open(newline="")).fieldnames)
    pkt = out_dir / f"irr_packet_{rater}"
    if pkt.exists():
        shutil.rmtree(pkt)
    (pkt / "frames").mkdir(parents=True)
    (pkt / "labels").mkdir()

    missing = []
    for r in rows:
        src = IMAGES / f"{r['frame_id']}.png"
        if src.exists():
            shutil.copy2(src, pkt / "frames" / src.name)
        else:
            missing.append(r["frame_id"])
    if missing:
        raise FileNotFoundError(f"missing images for: {missing}")

    with (pkt / "irr_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(blank_manifest_rows(rows, fieldnames))

    (pkt / "labeler.ipynb").write_text(json.dumps(build_notebook(rater, len(rows)), indent=1))
    (pkt / "README.md").write_text(_readme_source(rater, len(rows)))
    (pkt / "labels" / ".gitkeep").write_text("")

    zip_path = out_dir / f"irr_packet_{rater}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(pkt.rglob("*")):
            zf.write(p, p.relative_to(out_dir))
    print(f"  {rater}: {len(rows)} frames -> {zip_path.name} "
          f"({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raters", nargs="+", help="rater ids / initials, e.g. LG MS")
    ap.add_argument("--slots", type=int, default=SLOTS_PER_DAY,
                    help=f"max frames per day (default {SLOTS_PER_DAY})")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sel = select_frames(args.slots)
    print(f"IRR frame set: {len(sel)} frames, {args.slots} slots/day max")
    for rater in args.raters:
        build_packet(rater, args.slots, args.out)


if __name__ == "__main__":
    main()
