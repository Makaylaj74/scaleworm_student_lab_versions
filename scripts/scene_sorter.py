"""File-moving logic behind the Scene 1 / not-Scene 1 sorting widget.

A "session directory" is one output tree produced by ``scene_sampler`` and has
this shape::

    <session>/contact_sheets/CAMHDA301-*.png   # unsorted sheets
    <session>/scene1/                           # sorted: Scene 1
    <session>/not_scene1/                       # sorted: everything else
    <session>/sort_log.csv                      # append-only decision log

The notebook widget (``notebooks/25_sort_scenes.ipynb``) is a thin UI over these
functions; the decision/move/log logic lives here so it can be unit-tested.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

#: Terminal decisions that move a sheet out of ``contact_sheets/``.
FILED_DECISIONS: tuple[str, ...] = ("scene1", "not_scene1")
#: All decisions the UI may log ("skip" is logged but does not move a file).
DECISIONS: tuple[str, ...] = (*FILED_DECISIONS, "skip")


@dataclass(frozen=True)
class SessionPaths:
    """Resolved sub-paths for one sorting session."""

    root: Path
    contact_sheets: Path
    scene1: Path
    not_scene1: Path
    log_csv: Path

    def target_for(self, decision: str) -> Path:
        if decision == "scene1":
            return self.scene1
        if decision == "not_scene1":
            return self.not_scene1
        raise ValueError(f"decision {decision!r} does not map to a folder")


def session_paths(session_dir: Path) -> SessionPaths:
    """Resolve (and create) the standard sub-paths under a session directory."""
    root = Path(session_dir)
    paths = SessionPaths(
        root=root,
        contact_sheets=root / "contact_sheets",
        scene1=root / "scene1",
        not_scene1=root / "not_scene1",
        log_csv=root / "sort_log.csv",
    )
    for d in (paths.contact_sheets, paths.scene1, paths.not_scene1):
        d.mkdir(parents=True, exist_ok=True)
    return paths


def pending_sheets(session_dir: Path) -> list[Path]:
    """Return unsorted contact sheets (PNGs still in ``contact_sheets/``), sorted."""
    paths = session_paths(session_dir)
    return sorted(paths.contact_sheets.glob("*.png"))


def counts(session_dir: Path) -> dict[str, int]:
    """Return counts of sorted and pending sheets for progress display."""
    paths = session_paths(session_dir)
    return {
        "scene1": len(list(paths.scene1.glob("*.png"))),
        "not_scene1": len(list(paths.not_scene1.glob("*.png"))),
        "pending": len(list(paths.contact_sheets.glob("*.png"))),
    }


def _append_log(log_csv: Path, stem: str, decision: str, when: str) -> None:
    new_file = not log_csv.exists()
    with log_csv.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["stem", "decision", "decided_at_utc"])
        writer.writerow([stem, decision, when])


def apply_decision(
    session_dir: Path, sheet_path: Path, decision: str, when: str | None = None
) -> Path | None:
    """Record a sort decision, moving the sheet for terminal decisions.

    For ``scene1``/``not_scene1`` the sheet is moved into the matching folder
    (overwriting any same-named file there) and the move destination is returned.
    For ``skip`` the sheet is left in place and None is returned. Every decision
    is appended to ``sort_log.csv``.

    Raises:
        ValueError: if ``decision`` is not one of :data:`DECISIONS`.
        FileNotFoundError: if ``sheet_path`` does not exist.
    """
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
    sheet_path = Path(sheet_path)
    if not sheet_path.is_file():
        raise FileNotFoundError(sheet_path)

    paths = session_paths(session_dir)
    stamp = when or datetime.now(UTC).isoformat(timespec="seconds")
    _append_log(paths.log_csv, sheet_path.stem, decision, stamp)

    if decision == "skip":
        return None

    dest = paths.target_for(decision) / sheet_path.name
    if dest.exists():
        dest.unlink()
    shutil.move(str(sheet_path), str(dest))
    return dest
