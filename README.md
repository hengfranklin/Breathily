# Breathily

Breathily was a startup developing a contactless device to make **lung function assessment accessible to patients with physical disabilities** such as ALS and other neuromuscular diseases. Traditional spirometers require patients to form a tight seal around a mouthpiece and exhale forcefully — something many ALS patients cannot do due to facial muscle weakness and limited mobility. Breathily replaces the mouthpiece with **depth sensors and computer vision**, estimating lung function from chest wall movement alone.

> **Status:** Project ended 3/15/2022. This repo is an archive of the research code.

## Demo

[![Breathily Demo](https://img.youtube.com/vi/MaBf3D1GvQA/0.jpg)](https://www.youtube.com/watch?v=MaBf3D1GvQA)

▶️ Watch the demo: https://www.youtube.com/watch?v=MaBf3D1GvQA

## Recognition

- 🥈 **2nd Place**, UC Launch Accelerator Program
- 💰 Funding & support from the **UCSF Catalyst Program**
- 🔬 **NSF I-Corps** participant

## How it works

Breathily uses one or more **Intel RealSense** depth cameras pointed at a seated patient's torso. As the patient breathes, the system:

1. Detects the patient and locates the chest region using **skeleton tracking** (Cubemos).
2. Extracts a depth signal of chest wall displacement over time.
3. Identifies key respiratory phases (tidal breathing, start of exhale, end of exhale) via peak detection on the displacement curve.
4. Maps the depth-derived FVC to a calibrated lung volume using a regression model trained against ground-truth spirometry.
5. Computes standard pulmonary function test (PFT) measures: **FVC, FEV1, FEV1/FVC, PEF, FEF25, FEF50, FEF75, FEF25-75**.

## Repository layout

### [`helper_utils/`](helper_utils/) — runtime library

- [`realsense_manager.py`](helper_utils/realsense_manager.py) — `DeviceManager` class wrapping the Intel RealSense pipeline (depth + RGB streaming, alignment, post-processing filters, IR emitter control, playback from `.bag` files).
- [`skeleton_tracking.py`](helper_utils/skeleton_tracking.py) — Cubemos-based 2D/3D skeleton tracking utilities used to localize the chest ROI on the patient.
- [`patient_measurement.py`](helper_utils/patient_measurement.py) — `DeviceLungMeasure`, the top-level capture class that orchestrates a measurement session: streaming, ROI tracking, signal recording, and saving runs to disk.
- [`lung_measurement.py`](helper_utils/lung_measurement.py) — Signal processing and PFT computation. Detects respiratory keypoints, translates chest displacement to predicted lung volume via a saved regression model, and computes the full set of PFT metrics.
- [`user_control.py`](helper_utils/user_control.py) — Keyboard/UI control helpers for live capture sessions.
- [`DeviceManager.py`](helper_utils/DeviceManager.py) — placeholder.

### [`ipython_notebooks/`](ipython_notebooks/) — research & pipeline notebooks

- [`pipeline_master_v1.ipynb`](ipython_notebooks/pipeline_master_v1.ipynb) / [`pipeline_master_v2.ipynb`](ipython_notebooks/pipeline_master_v2.ipynb) — End-to-end measurement pipelines: capture → chest tracking → signal extraction → PFT computation. v2 is the latest iteration.
- [`reading_chest_vol_master.ipynb`](ipython_notebooks/reading_chest_vol_master.ipynb) — Reads recorded RealSense `.bag` files and extracts chest volume / displacement signals.
- [`compute_lung_params_master.ipynb`](ipython_notebooks/compute_lung_params_master.ipynb) — Computes PFT parameters (FVC, FEV1, PEF, FEF series) from chest displacement signals and compares against spirometer ground truth.
- [`dual_depth_vol_master.ipynb`](ipython_notebooks/dual_depth_vol_master.ipynb) — Experiments using **two depth cameras** to reconstruct chest volume more accurately than a single front-facing sensor.
- [`realtime_chest_visualization_master.ipynb`](ipython_notebooks/realtime_chest_visualization_master.ipynb) — Live visualization of the chest displacement signal during a capture session.
- [`skeleton_tracking_master.ipynb`](ipython_notebooks/skeleton_tracking_master.ipynb) — Standalone skeleton tracking development and debugging.

## Dependencies

- Python 3
- `pyrealsense2` (Intel RealSense SDK)
- Cubemos Skeleton Tracking SDK
- `numpy`, `scipy`, `pandas`, `scikit-learn`, `joblib`, `scikit-image`, `opencv-python`, `matplotlib`

## Contact

Franklin Heng — heng.franklin@berkeley.edu
