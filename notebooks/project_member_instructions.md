# Scaleworm Notebook — Project Member Instructions

Notebook: `22_verify_detections.ipynb`
Last updated: 2026-06-28

---

## Option A — Starting from Ultralytics Hub annotations (Sections 9–10 only)

Use this if the frame extraction, detection, and verification have already been done
and your role is to import corrected bounding box annotations and retrain the model.

**What you need before starting:**
- The notebook open in JupyterLab with the correct kernel selected
- Your `.ndjson` file downloaded from Ultralytics Hub

**Step 1 — Run the Setup cell (Section 1)**

Run the Setup cell first. It initialises all the path variables the later cells depend on.
You do not need to change anything. Confirm the output shows `Exists: True` for both
the model path and video root before continuing.

**Step 2 — Run Section 9 (Import corrected annotations)**

Find this line in the cell and update it to point to your `.ndjson` file:

    NDJSON_PATH = Path("/home/jovyan/your_name/round_2.ndjson")

Leave `IMAGES_DIR` and `CORRECTED_DIR` as-is and run the cell.

Check the output:
- `Images with annotations` and `Total bounding boxes` should be in the thousands
- If either is 0, or you see many "image not found" warnings, double-check that
  `NDJSON_PATH` points to the correct file

**Step 3 — Run Section 10 (Retrain)**

Default settings (`EPOCHS = 20`, `IMGSZ = 1280`) are fine. Run the cell.
Training takes roughly 20–60 minutes on CPU, faster with GPU.

When it finishes the output will show the exact path to the new `best.pt` weights file.

**Step 4 — Record your mAP50**

The training log prints a `mAP50` value at the end. Write this down — it is the
accuracy metric being tracked across rounds.

**You do not need to run:** Sections 2–8. Skip them entirely.

---

## Option B — Running the full pipeline from scratch (Sections 1–10)

Use this if you are starting with fresh video data and doing your own detections
and verification from the beginning.

**What you need before starting:**
- Access to the CAMHD video archive at `/home/jovyan/ooi/san_data/RS03ASHS-PN03B-06-CAMHDA301/`
- The notebook open in JupyterLab with the correct kernel selected
- A Ultralytics Hub account (free) for the bounding box correction step

**Step 1 — Run Section 1 (Setup)**

Run the Setup cell. Confirm `Exists: True` for both the model path and video root.
If the model path does not exist, ask for the correct path to `best.pt`.

**Step 2 — Run Section 2 (Choose your date range)**

Set `START_DATE` and `END_DATE` to the month you want to analyse, e.g.:

    START_DATE = "2024-10-01"
    END_DATE   = "2024-10-31"

Run the cell and confirm it finds the expected number of videos.

**Step 3 — Run Section 3 (Extract frames)**

This pulls one frame per second from the 305–320 second window of each video.
It is resumable — already-extracted frames are skipped automatically.
Expect this to take 15–30 minutes for a full month.

**Step 4 — Run Section 4 (Run the YOLO detector)**

Run the `pip install` cell first, then the detector cell.
The detector runs at a low confidence threshold (0.1) to catch as many candidates
as possible — expect many false positives, which you will sort in the next step.
On CPU this takes 30–60 minutes for a full month.

After the detector finishes, run the save cell immediately (the one that prints
"Saved N detections"). Do not skip this.

**Step 4b — Filter detections (if count is above ~3000)**

If the total detection count is unexpectedly high (above ~3000 for a full month),
run the filter cell to cut low-confidence detections before cropping:

    FILTER_CONF = 0.3

This is normal behaviour for a retrained model and does not indicate an error.

**Step 5 — Run Section 5 (Crop detections)**

This saves a cropped image of each detection for review. Resumable — skip if
crops already exist. Takes a few minutes.

**Step 6 — Run Section 6 (Verify detections)**

This is the main task. For each crop you will see the detection at 1x, 2x, and 4x zoom.
Click one of three buttons:

- Scale Worm — confirmed true detection
- Not a Worm — false positive (tube, bacterial mat, artifact, etc.)
- Skip — genuinely unsure

Your progress is saved after every click. You can close the notebook and resume
later — re-run Sections 1–5 (they will skip already-completed work) and the widget
will pick up where you left off.

**Step 7 — Run Section 7 (Summary)**

Check that `Unlabeled: 0` before continuing. If unlabeled detections remain,
go back to Section 6.

**Step 8 — Run Section 8 (Export)**

This packages your verified detections as a YOLO dataset and saves a zip file.
Note the path to the zip — you will upload it to Ultralytics Hub next.

**Step 9 — Correct bounding boxes in Ultralytics Hub**

1. Upload the zip from Step 8 to Ultralytics Hub
2. Review and correct the bounding boxes on each image
3. Export the corrected annotations as `.ndjson`
4. Download the `.ndjson` file to your JupyterHub home directory

**Step 10 — Run Section 9 (Import corrected annotations)**

Update `NDJSON_PATH` to point to your downloaded `.ndjson` file and run the cell.
Confirm the image and bounding box counts look correct.

**Step 11 — Run Section 10 (Retrain)**

Run the retraining cell. When it finishes, record the `mAP50` value from the
training log and note the path to the new `best.pt` weights file.

---

## Notes for all users

- The notebook saves progress automatically. If your session disconnects, re-run
  from the top — completed steps skip automatically.
- Do not run more than one detection or retraining cell at the same time.
- If you see a CUDA warning about the NVIDIA driver, ignore it. The notebook
  runs on CPU and produces correct results, just more slowly.
- Questions? Contact the project lead before modifying any configuration constants.
