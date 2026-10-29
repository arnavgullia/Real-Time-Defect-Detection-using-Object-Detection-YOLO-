# Real-Time Industrial Defect Detection System Using YOLOv8

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, production-grade computer vision system designed for real-time automated surface defect detection and localization on hot-rolled steel strips. Built using the anchor-free YOLOv8 architecture, this system supports instant terminal image inspection, batch inference, continuous video detection, live webcam monitoring, COCO-scale defect analytics, and comprehensive research documentation.

Developed for **Project-Based Learning (PBL-I)** university review.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INDUSTRIAL DEFECT SYSTEM                            │
├───────────────────────┬──────────────────────────┬──────────────────────────┤
│    DATA PIPELINE      │     TRAINING & EVAL      │   INSPECTION INTERFACE   │
│ 1,800 NEU-DET Images  │ YOLOv8n (Anchor-Free)    │ Terminal Defect CLI      │
│ Pascal VOC XML Parser │ TaskAlignedAssigner Head │ Direct OpenCV Streamer   │
│ Stratified Splits     │ COCO-Scale Metrics       │ Structured JSON Outputs  │
│ Auto BBox Clamping    │ FPS Latency Profiler     │ Real-time Telemetry HUD  │
└───────────────────────┴──────────────────────────┴──────────────────────────┘
```

---

## 2. Dataset Overview: NEU-DET

The system is calibrated on the benchmark **NEU Surface Defect Database (NEU-DET)** from Northeastern University:
- **Total Images**: 1,800 uniform 200 × 200 pixel images (3-channel RGB).
- **Total Defect Instances**: 4,189 verified bounding boxes.
- **Defect Categories (6 Classes)**:
  1. `crazing`: Web-like surface micro-cracks (689 instances).
  2. `inclusion`: Non-metallic embedded impurities (1,011 instances).
  3. `patches`: Surface oxidation blotches (881 instances).
  4. `pitted_surface`: Cavity depressions and micro-pits (432 instances).
  5. `rolled-in_scale`: Surface iron oxide rolling streaks (628 instances).
  6. `scratches`: Longitudinal linear mechanical abrasions (548 instances).
- **Stratified Splits**:
  - **Train**: 1,440 images (80%, 3,335 defect boxes).
  - **Validation**: 180 images (10%, 441 defect boxes).
  - **Test (Held-Out)**: 180 images (10%, 413 defect boxes).

---

## 3. Directory Layout

```
APPLIED AI 2/
├── raw_dataset/                    # Original uncorrupted VOC dataset (read-only)
├── datasets/
│   └── defect_dataset/             # Prepared YOLOv8 dataset
│       ├── images/ [train, val, test]
│       ├── labels/ [train, val, test]
│       ├── class_mapping.json
│       └── data.yaml
├── src/
│   ├── prepare_dataset.py          # VOC XML to YOLO converter with stem indexing
│   ├── validate_dataset.py         # Automated label & image integrity verifier
│   ├── analyze_dataset.py          # COCO-scale EDA and chart generator
│   ├── train_baseline.py           # Reproducible YOLOv8 baseline trainer
│   ├── evaluate.py                 # Latency profiling & mAP evaluator
│   ├── predict.py                  # Batch and single-image inference
│   ├── video_detect.py             # Live webcam / video OpenCV streamer
│   └── utils.py                    # BBox math, logging, and color palettes
├── detect_defect.py                # Instant terminal defect inspection CLI
├── configs/
│   └── experiments.yaml            # Hyperparameter & anchor benchmark matrix
├── outputs/
│   ├── baseline/                   # Trained weights (best.pt) and run logs
│   ├── predictions/                # Output detection overlays and JSON logs
│   └── reports/                    # Validation, EDA, and evaluation reports
│       ├── annotation_checks/      # Visual ground-truth verification samples
│       └── eda/                    # 6 publication charts and 5 summary CSVs
├── research/
│   ├── dataset_documentation.md    # Provenance, taxonomy, and normalization
│   ├── research_logbook.md         # Chronological log & actual debugging events
│   ├── experiment_plan.md          # Variable controls & anchor benchmark protocol
│   └── results.md                  # Measured metrics, FPS, and per-class AP
├── tests/
│   └── test_pipeline.py            # Automated pytest unit test suite
├── requirements.txt                # Pinned dependencies
├── run_project.ps1                 # Master Windows PowerShell execution script
├── DATASET_STATUS.md               # Dataset audit and discrepancy sign-off
└── README.md                       # Complete documentation
```

---

## 4. Quickstart Guide (Windows PowerShell)

Open PowerShell in the workspace root (`c:\Users\arnav\OneDrive\Desktop\APPLIED AI 2`):

### 1. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Prepare Dataset (XML to YOLO Conversion)
```powershell
python src/prepare_dataset.py
```

### 4. Validate Converted Dataset
```powershell
python src/validate_dataset.py
```

### 5. Generate Exploratory Data Analysis & Charts
```powershell
python src/analyze_dataset.py
```

### 6. Run Automated Test Suite
```powershell
python -m pytest tests/test_pipeline.py -v
```

### 7. Train Baseline Model (Configurable Epochs / Imgsz)
```powershell
# Quick CPU benchmark (1 epoch)
python src/train_baseline.py --epochs 1 --imgsz 320 --batch 16

# Full standard baseline training (50 epochs)
python src/train_baseline.py --epochs 50 --imgsz 640 --batch 16
```

### 8. Evaluate Trained Model
```powershell
python src/evaluate.py --weights outputs/baseline/best.pt --split val
```

### 9. Run Single-Image or Batch Inference
```powershell
python src/predict.py --source datasets/defect_dataset/images/test --weights outputs/baseline/best.pt
```

### 10. Real-Time Video or Live Webcam Stream
```powershell
# Live USB Webcam (Device 0)
python src/video_detect.py --source 0 --weights outputs/baseline/best.pt

# Video File Inspection
python src/video_detect.py --source path\to\steel_roll.mp4 --weights outputs/baseline/best.pt --output outputs\predictions\annotated_roll.mp4
```

### 11. Instant Terminal Defect Detection (CLI)
```powershell
# Test any custom image directly in terminal:
python detect_defect.py path/to/image.jpg

# Test with custom confidence threshold (e.g. 0.20):
python detect_defect.py path/to/image.jpg 0.20

# Test a random sample from the test set:
python detect_defect.py
```

---

## 5. Measured Baseline Performance

Measured on **AMD Ryzen 7 7435HS** (Host CPU Execution, AMD64):

| Metric | Measured Baseline |
| :--- | :---: |
| **Model Architecture** | YOLOv8n (3.13M parameters, 8.6 GFLOPs) |
| **Model Size** | 6.5 MB (`outputs/baseline/best.pt`) |
| **Host CPU Inference Latency** | **15.6 ms** per image (64 FPS throughput) |
| **Preprocess + Postprocess** | 0.4 ms + 2.0 ms |
| **Hardware Mode** | Host CPU (Zero GPU required for real-time video) |
| **Pitted Surface AP@50** | **11.6%** (Early convergence) |

*Full metrics, faster-coco-eval area breakdowns, and per-class AP tables are documented in `research/results.md`.*

---

## 6. Terminal Detection CLI Output

When running `python detect_defect.py <image>`, the detector formats human-readable anomaly telemetry directly in the shell:

```
=======================================================
  SURFACE INSPECTION: patches_241.jpg
=======================================================
  DECISION : [REJECT] - 2 Surface Defect(s) Detected!
  Latency  : 16.5 ms
  -----------------------------------------------------
  Detected Anomalies:
   1. Class: patches          | Confidence:  36.3% | BBox: [71, 0, 134, 188]
   2. Class: patches          | Confidence:  16.3% | BBox: [72, 48, 120, 106]
  Annotated: outputs/predictions/pred_patches_241.jpg
=======================================================
```

---

## 7. Troubleshooting & Engineering Notes

1. **Missing Trained Weights Error**:  
   If `detect_defect.py` or `predict.py` warns that `best.pt` is missing, run `python src/train_baseline.py` to train the custom detector.
2. **Webcam Device Permissions**:  
   If `video_detect.py` cannot access camera index 0, verify that Windows Privacy Settings allow desktop apps to access the camera, or pass a recorded video file via `--source sample.mp4`.
3. **Anchor Tuning Note**:  
   YOLOv8 is an anchor-free detector using the Task-Aligned Assigner. Anchor tuning experiments are benchmarked via a parallel YOLOv5n protocol detailed in `research/experiment_plan.md`.

