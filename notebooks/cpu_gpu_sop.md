# CPU vs GPU Standard Operating Procedure

Scaleworm Student Lab — last updated 2026-07-02

---

## Why this matters

Most pipeline steps run fine on CPU. Two steps — YOLO detection and model retraining —
benefit strongly from a GPU and become significantly slower without one. This document tells
you which is which, how to check what your machine has, and what to do in each case.

---

## Quick reference

| Pipeline step | Notebook section | CPU or GPU? | Typical CPU time | Typical GPU time |
|---|---|---|---|---|
| Setup / path init | Section 1 | CPU only | < 1 min | — |
| Choose date range | Section 2 | CPU only | < 1 min | — |
| Frame extraction (ffmpeg) | Section 3 | CPU only | 15–30 min / month | — |
| **YOLO detection inference** | **Section 4** | **GPU-capable** | **30–60 min / month** | **5–15 min / month** |
| Filter detections | Section 4b | CPU only | < 1 min | — |
| Crop detections | Section 5 | CPU only | 2–5 min | — |
| Human verification (widget) | Section 6 | CPU only | 2–3 hours (human pace) | — |
| Summary check | Section 7 | CPU only | < 1 min | — |
| Export / packaging | Section 8 | CPU only | < 1 min | — |
| Import corrected annotations | Section 9 | CPU only | < 1 min | — |
| **Model retraining** | **Section 10** | **GPU-capable** | **20–60 min** | **5–15 min** |

**GPU-capable** steps run on CPU without any code changes — they are just slower.
Everything else is CPU-only by nature and gains nothing from a GPU.

---

## How to check whether your session has a GPU

Run this cell before starting any GPU-capable step:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

- If `CUDA available: True` — you have a GPU; the notebook will use it automatically.
- If `CUDA available: False` — you are on CPU; everything still works, just slower.

**OOI JupyterHub note:** The Hub does not currently have a GPU. All Hub sessions run on CPU.
The runtimes in the "Typical CPU time" column above are what Hub users should expect.

---

## What to do based on your situation

### On the OOI JupyterHub (CPU only)
- Run all steps normally. The notebook handles CPU automatically.
- Detection (Section 4) and retraining (Section 10) will take the longer CPU times listed above.
- Do not start a detection run and close the tab — **keep the browser tab active** for the full
  run, or use the resumable shortcut in the instructions if your session drops.
- **Never run Section 4 and Section 10 at the same time** — they compete for CPU and will
  slow each other significantly.

### On a local machine with an NVIDIA GPU
- Make sure the `scaleworm` conda environment is active and that your PyTorch was installed
  with CUDA support (`torch.cuda.is_available()` returns `True`).
- No code changes needed — YOLO and PyTorch detect the GPU automatically.
- GPU runs are 4–6× faster for inference and retraining.

### On a local machine without a GPU (Mac, older PC)
- Same as the Hub: runs on CPU, normal timings apply.

---

## Rules for GPU-capable steps

1. **One GPU job at a time per machine.** Do not run two detection or retraining cells
   simultaneously — they will fight for VRAM and both will fail or stall.
2. **Check GPU memory before retraining.** If you get a CUDA out-of-memory error, reduce
   `IMGSZ` from 1280 to 640 in the retraining cell and re-run.
3. **Do not mix environments.** If you installed PyTorch with GPU support in the `scaleworm`
   environment, do not run GPU steps in the base kernel — it will silently fall back to CPU.
4. **Record which device was used.** When you log your `mAP50` at the end of retraining,
   also note whether training ran on CPU or GPU. This matters when comparing run times across
   rounds.

---

## Common questions

**"I see a CUDA warning about the NVIDIA driver — should I worry?"**
No. On the Hub this warning appears because CUDA is not available. The notebook detects
this and falls back to CPU automatically. Ignore the warning and continue.

**"My detection run is at 20% after an hour — is something wrong?"**
No, this is normal on CPU for a large month with many videos. Let it run. If the browser
tab is still connected and the cell is still executing, it is still working. Use the
resume shortcut (re-run Section 1 then Section 4) if the session drops.

**"Can I use Google Colab for the GPU steps?"**
Ask your advisor before moving data off the Hub. The video archive and model weights are
stored on the Hub's shared filesystem and are not available outside it.

---

*Questions about hardware access or GPU availability? Contact the project lead.*
