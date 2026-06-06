# Scaleworm Student Lab — Claude Code Instructions

## Project Context
Educational platform for human-in-the-loop object detection. Students train Faster R-CNN v2 models on OOI CAMHD video to count scale worms at Mushroom vent (ASHES, Axial Seamount). Designed as a hands-on lab exercise.

## Package Management
- Uses `conda` environment with PyTorch and ffmpeg
- `pyproject.toml` for metadata

## Code Conventions
- Student notebook: `notebooks/20_student_scaleworm_pipeline.ipynb` (monolithic) or split into `20a_student_setup.ipynb` + `20b_student_labeling_loop.ipynb`
- Instructor comparison: `notebooks/21_cross_student_comparison.ipynb`
- Build scripts in `scripts/`: `build_student_notebook.py`, `build_starter_package.py`, `build_comparison_notebook.py`

## Student Workflow (4 phases)
1. **Phase 1**: Run pretrained detector (~30 min)
2. **Phase 2**: Review & correct mistakes (2–3 hours)
3. **Phase 3**: Retrain model (~1 hour)
4. **Phase 4**: Export final counts & quality report

## Scientific Standards
- Starter package: `~/scaleworm_starter_package.zip` (~3 GB) with MP4 clips, VIAME CSVs, pretrained v3 model, ground truth counts
- Scenes: March, April, June, August 2023
- Baseline model: `best_model_v3.pth` (MAE=2.9, r=0.933)
- Ground truth: `scaleworm_counts.parquet`
- Colorblind-safe palettes (Okabe-Ito, viridis)

## Figure Standards
- Run the `timeseries-figure` skill rubric when reviewing student output plots
- All figures need justified captions

## Key Consideration
This is student-facing code — prioritize clarity, helpful error messages, and well-commented notebook cells over brevity. The audience is undergraduate/early-graduate students who may be new to Python and ML.
