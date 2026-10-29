# Research Logbook: Real-Time Industrial Defect Detection Using YOLOv8

**Project Title**: Real-Time Industrial Surface Defect Detection Using YOLOv8  
**Repository**: `APPLIED AI 2`  
**Host OS**: Windows 10/11 (PowerShell)  
**Hardware Profile**: AMD Ryzen 7 7435HS (8 Cores / 16 Threads), Host CPU Execution Mode (No CUDA GPU)  
**Primary Frameworks**: Ultralytics YOLOv8, PyTorch, Streamlit, OpenCV  

---

## Logbook Entries & Timeline

### Entry 1: Workspace Inspection & Provenance Audit
- **Date**: 2026-10-27
- **Activities**:
  - Conducted full recursive filesystem crawl on workspace `APPLIED AI 2`.
  - Discovered authentic raw dataset under `raw_dataset/` with `train` (1,440 images across 6 folders) and `validation` (360 images across 6 folders).
  - Verified image format: 200×200 pixel JPEGs, RGB 3-channel.
  - Verified annotation format: Pascal VOC XML with standard `<bndbox>` tags.
  - Formally established dataset provenance as **NEU-DET** (Northeastern University Surface Defect Database).
- **Decisions**:
  - Imposed strict read-only policy on `raw_dataset/` to preserve data integrity.
  - Documented findings in `DATASET_STATUS.md`.

---

### Entry 2: Debugging Event & Resolution — File Stem Pairing Discrepancy
- **Date**: 2026-10-27
- **Problem Encountered**:
  - Initial directory counts showed `raw_dataset/train/annotations/` containing 1,439 XML files while `train/images/` had 1,440 images.
  - In contrast, `raw_dataset/validation/annotations/` had 361 XML files while `validation/images/` had 360 images.
  - Immediate hypothesis: An image was missing an annotation or a file was misplaced across folders.
- **Root Cause Analysis**:
  - Ran automated set difference of file stems across both directories.
  - Found that `crazing_240.jpg` was located in `raw_dataset/train/images/crazing/crazing_240.jpg`, but its annotation `crazing_240.xml` was placed inside `raw_dataset/validation/annotations/crazing_240.xml`.
  - Global cross-check confirmed: exactly 1,800 unique images and 1,800 unique XML annotations exist across the workspace. Zero missing files.
- **Engineering Solution**:
  - In `src/prepare_dataset.py`, constructed a global index dictionary of all XML annotations keyed by lowercase stem name.
  - Bounded each image strictly to its genuine annotation without modifying or moving original files.
  - Assigned `crazing_240` to the `train` split alongside its matching image to maintain zero train/validation leakage.

---

### Entry 3: Environment Dependency Setup & Compatibility
- **Date**: 2026-10-27
- **Problem Encountered**:
  - Testing package availability revealed `ultralytics` and `streamlit` were missing from `.venv`.
- **Resolution**:
  - Executed `pip install ultralytics streamlit`.
  - Installed `ultralytics-8.4.190` and `streamlit-1.66.1`.
  - Addressed minor package dependency warnings regarding script path locations on Windows.
  - Verified runtime import through automated verification script.

---

### Entry 4: Pipeline Implementation — XML to YOLO Conversion & Validation
- **Date**: 2026-10-27
- **Activities**:
  - Built `src/prepare_dataset.py`:
    - Pascal VOC to YOLO normalization.
    - Preserved 1,440 train pool images (80%).
    - Deterministically partitioned 360 validation images 50/50 using fixed seed (`seed=42`) into 180 `val` (10%) and 180 `test` (10%) images with stratified class representation (30 images/class each).
  - Built `src/validate_dataset.py`:
    - Validated all 1,800 images and label files.
    - Verified 5-field lines, class IDs (0–5), coordinate boundedness, positive widths/heights.
    - Rendered 18 ground-truth visual overlays in `outputs/reports/annotation_checks/`.
    - **Validation Result**: 100% PASSED (0 errors).

---

### Entry 5: Debugging Event & Resolution — Tabulate Dependency in EDA
- **Date**: 2026-10-27
- **Problem Encountered**:
  - Execution of `src/analyze_dataset.py` threw `ImportError: Missing optional dependency 'tabulate'` when calling `df_scale_by_class.to_markdown()`.
- **Root Cause**:
  - Pandas `to_markdown()` depends on the optional third-party library `tabulate`.
- **Resolution**:
  - Refactored `src/analyze_dataset.py` to construct GitHub-flavored markdown tables natively using string formatting.
  - Re-executed `python src/analyze_dataset.py`.
  - Generated all 6 publication charts and 5 structured CSV files under `outputs/reports/eda/`.

---

### Entry 6: Baseline Training & Hardware Profiling
- **Date**: 2026-10-27
- **Hardware Profile Measured**:
  - Device: AMD Ryzen 7 7435HS (CPU mode, CUDA unavailable).
  - Ingestion rate: ~1.1 to 1.2 seconds per batch (batch size 16, imgsz=320).
  - 1 epoch runtime: ~1 minute 40 seconds.
- **Architectural Decision**:
  - YOLOv8 is anchor-free, utilizing the TaskAlignedAssigner head.
  - Baseline model selected: `yolov8n.pt` (3.01M parameters, 8.2 GFLOPs).
  - Full baseline training is runnable via `python src/train_baseline.py`.

---

## PBL-I Experiment Tracking Table

| Exp ID | Model Arch | Resolution | Batch | Device | Epochs | mAP@50 | mAP@50-95 | Latency (ms) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `EXP-01` | YOLOv8n (Baseline) | 640 | 16 | CPU/CUDA | 50 | *Pending Full Run* | *Pending* | *Pending* | Configured & Ready |
| `EXP-02` | YOLOv8n (Edge Res) | 320 | 16 | CPU | 1 | Benchmarking | Benchmarking | ~18 ms | Completed Test Run |
| `EXP-03` | YOLOv8n (Fine Res) | 960 | 8 | CUDA | 40 | *Planned* | *Planned* | *Planned* | In Plan |
| `EXP-04` | YOLOv8s (Deeper) | 640 | 8 | CUDA | 40 | *Planned* | *Planned* | *Planned* | In Plan |
| `EXP-05` | YOLOv8n (Defect Aug) | 640 | 16 | CUDA | 40 | *Planned* | *Planned* | *Planned* | In Plan |
| `EXP-06` | YOLOv5n (Default Anchor) | 640 | 16 | CUDA | 40 | *Planned* | *Planned* | *Planned* | In Plan |
| `EXP-07` | YOLOv5n (Custom Anchor) | 640 | 16 | CUDA | 40 | *Planned* | *Planned* | *Planned* | In Plan |

*(Table updated dynamically as training runs execute)*
