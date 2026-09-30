"""Recording-level gate for the Scene-1 tile classifier.

The tile-level test metrics (see train_scene_classifier.py) say the model is strong,
but the operational question is: run over EVERY tile of a fresh recording, does it
make the same Scene-1/not call a human would? This script answers that against a
blind human sort.

For each recording in a sort session (default the blind 2016 subsample):
  1. sample tiles on the standard 30 s grid (0..840 s), read each frame with cv2,
  2. score them with the classifier,
  3. aggregate to a recording-level call: Scene-1 iff the top-scoring tile clears the
     model's threshold; predicted time = the argmax tile (matching how inference will
     pick a frame to count).

Writes predictions.csv immediately (needs no human labels). If the session's
sort_log.csv already holds human decisions, it ALSO scores the gate: raw agreement
(Wilson 95% CI), Cohen's kappa, the disagreement list, and — for recordings both call
Scene-1 — whether the predicted time lands within one 30 s tile of the human's. Result
prose goes to gate_results.md with the lab AI-disclosure label.

No sklearn (unavailable here): kappa / Wilson / CI are implemented directly.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parent.parent
DEF_SESSION = REPO / "scene_sorting/subsample_2016"
DEF_MODEL = REPO / "models/scene_classifier/best.pt"
GRID = list(range(0, 841, 30))  # 0..840 s; t=0 included (a real 2015 Scene-1 sat there)


def load_model(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from train_scene_classifier import SmallCNN

    model = SmallCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tf = transforms.Compose(
        [
            transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return model, tf, float(ckpt["threshold"])


def score_recording(video: Path, model, tf, grid=GRID) -> tuple[list[int], np.ndarray]:
    """Return (tile_times_with_frames, per-tile Scene-1 probabilities)."""
    cap = cv2.VideoCapture(str(video))
    tensors, times = [], []
    for t in grid:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue  # tile past end of a short pan
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensors.append(tf(img))
        times.append(t)
    cap.release()
    if not tensors:
        return [], np.array([])
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.stack(tensors))).numpy()
    return times, probs


def predict_session(session: Path, model, tf, thr: float) -> list[dict]:
    manifest = session / "manifest.csv"
    rows = list(csv.DictReader(manifest.open()))
    out = []
    for i, r in enumerate(rows, 1):
        times, probs = score_recording(Path(r["video_path"]), model, tf)
        if len(probs) == 0:
            out.append({"stem": r["stem"], "pred_decision": "no_frames",
                        "pred_scene1_time_s": "", "max_prob": ""})
            continue
        j = int(probs.argmax())
        out.append({
            "stem": r["stem"],
            "pred_decision": "scene1" if probs[j] >= thr else "not_scene1",
            "pred_scene1_time_s": times[j] if probs[j] >= thr else "",
            "max_prob": round(float(probs[j]), 4),
        })
        print(f"  [{i}/{len(rows)}] {r['stem']} -> {out[-1]['pred_decision']} "
              f"(p={out[-1]['max_prob']})")
    return out


# ---- stats (no sklearn) ----------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    labels = ["scene1", "not_scene1"]
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum(
        (a.count(lb) / n) * (b.count(lb) / n) for lb in labels
    )
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def score_gate(preds: list[dict], session: Path) -> dict | None:
    log = session / "sort_log.csv"
    if not log.exists():
        return None
    human = {}
    for r in csv.DictReader(log.open()):
        if r["decision"] in ("scene1", "not_scene1"):
            human[r["stem"]] = r
    pred_by = {p["stem"]: p for p in preds}
    paired = [(s, human[s], pred_by[s]) for s in human if s in pred_by
              and pred_by[s]["pred_decision"] in ("scene1", "not_scene1")]
    if not paired:
        return {"n_paired": 0}
    h = [hp["decision"] for _, hp, _ in paired]
    m = [pp["pred_decision"] for _, _, pp in paired]
    agree = sum(x == y for x, y in zip(h, m))
    n = len(paired)
    lo, hi = wilson_ci(agree, n)
    disagreements = [
        {"stem": s, "human": hp["decision"], "model": pp["pred_decision"],
         "model_p": pp["max_prob"]}
        for s, hp, pp in paired if hp["decision"] != pp["pred_decision"]
    ]
    # timing agreement on mutual Scene-1
    timing = []
    for s, hp, pp in paired:
        if hp["decision"] == "scene1" == pp["pred_decision"] and hp["scene1_time_s"]:
            dt = abs(int(pp["pred_scene1_time_s"]) - int(float(hp["scene1_time_s"])))
            timing.append({"stem": s, "human_t": int(float(hp["scene1_time_s"])),
                           "model_t": int(pp["pred_scene1_time_s"]), "abs_dt_s": dt})
    within1 = sum(t["abs_dt_s"] <= 30 for t in timing)
    return {
        "n_paired": n, "agreement": agree / n, "agree_ci95": [lo, hi],
        "cohen_kappa": cohen_kappa(h, m),
        "n_human_scene1": h.count("scene1"), "n_model_scene1": m.count("scene1"),
        "disagreements": disagreements,
        "timing_n": len(timing), "timing_within_1tile": within1, "timing_detail": timing,
    }


def write_report(gate: dict, out_dir: Path) -> Path:
    """gate_results.md with the lab AI-disclosure label (prose intended for humans)."""
    p = out_dir / "gate_results.md"
    k = gate.get("cohen_kappa", float("nan"))
    verdict = ("PASS — strong agreement" if k >= 0.6 else
               "MARGINAL" if k >= 0.4 else "FAIL — agreement no better than chance")
    lo, hi = gate["agree_ci95"]
    disclosure = (
        "*AI-generated draft (Claude, Anthropic) — for review. All numbers are computed "
        "by `scripts/gate_scene_classifier.py` from version-controlled labels and the "
        "trained model.*"
    )
    agreement_line = (
        f"- Raw agreement: **{gate['agreement']:.1%}** "
        f"(Wilson 95% CI {lo:.1%}–{hi:.1%})"
    )
    base_rate_line = (
        f"- Human called Scene-1 on {gate['n_human_scene1']}; "
        f"model called Scene-1 on {gate['n_model_scene1']}"
    )
    lines = [
        disclosure,
        "",
        "# Scene-classifier gate — blind 2016 subsample",
        "",
        f"**Verdict: {verdict}** (Cohen's kappa = {k:.3f}).",
        "",
        f"- Paired recordings scored: **{gate['n_paired']}**",
        agreement_line,
        f"- Cohen's kappa: **{k:.3f}**",
        base_rate_line,
        f"- Disagreements: **{len(gate['disagreements'])}**",
    ]
    for d in gate["disagreements"]:
        lines.append(
            f"    - `{d['stem']}`: human={d['human']} model={d['model']} "
            f"(p={d['model_p']})"
        )
    timing_line = (
        f"- Timing (recordings both call Scene-1, n={gate['timing_n']}): "
        f"**{gate['timing_within_1tile']}/{gate['timing_n']}** within one 30 s tile of "
        "the human's marked time."
    )
    caveat_line = (
        "Caveats (Defensible Statistics): n is small (a 24-sheet subsample), so the "
        "agreement CI is wide and kappa is sensitive to the Scene-1/not base rate. This "
        "gate tests the 2016 (clear-era) transfer specifically; post-2023 blurry footage "
        "is weaker (see metrics.json). The count target for the early era is a separate "
        "open decision (advisor)."
    )
    lines += ["", timing_line, "", caveat_line]
    p.write_text("\n".join(lines) + "\n")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", type=Path, default=DEF_SESSION)
    ap.add_argument("--model", type=Path, default=DEF_MODEL)
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default validation/scene_classifier_gate/<session>)")
    args = ap.parse_args(argv)

    out_dir = args.out or (REPO / "validation/scene_classifier_gate" / args.session.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tf, thr = load_model(args.model)
    print(f"model threshold={thr:.3f}; scoring {args.session.name} ...")
    preds = predict_session(args.session, model, tf, thr)
    pred_csv = out_dir / "predictions.csv"
    with pred_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(preds[0].keys()))
        w.writeheader()
        w.writerows(preds)
    n_s1 = sum(p["pred_decision"] == "scene1" for p in preds)
    print(f"wrote {pred_csv} ({len(preds)} recs, model Scene-1 on {n_s1})")

    gate = score_gate(preds, args.session)
    if gate is None or gate.get("n_paired", 0) == 0:
        print("No human labels yet — predictions cached. Sort the session blind in "
              "nb25, then re-run this to score the gate.")
        return 0
    import json

    (out_dir / "gate_metrics.json").write_text(json.dumps(gate, indent=2))
    report = write_report(gate, out_dir)
    print(f"GATE: kappa={gate['cohen_kappa']:.3f} agreement={gate['agreement']:.1%} "
          f"(n={gate['n_paired']})  ->  {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
