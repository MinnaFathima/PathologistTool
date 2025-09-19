# Interactive Pathologist Tool (Prototype)

## Overview
This repository provides a Streamlit-based prototype that wraps your Hybrid CNN+ViT PyTorch model and provides:
- Whole-image tiling and tile-level inference
- Grad-CAM heatmaps (from CNN branch) stitched into whole-image heatmaps
- Simple ViT attention rollout (utility)
- Streamlit UI to upload images, view heatmaps, add annotations (as JSON), and save to SQLite DB
- Active learning export script to convert saved annotated images into tile-level images for re-training

## Quick start
1. Install requirements: `pip install -r requirements.txt`
2. Place your trained PyTorch checkpoint in the working folder named `model_checkpoint.pth` (or edit `app.py` sidebar to point to path).
3. Run streamlit: `streamlit run app.py`
4. Upload an image and test inference. Save annotations, export CSV, and use `active_learning.py` to export tiles for retraining.

## Notes
- This is a prototype for research/testing only. Do not use clinically without rigorous validation.
- For production: store images in secure storage, switch to MongoDB, add authentication, and get clinician oversight.
```
