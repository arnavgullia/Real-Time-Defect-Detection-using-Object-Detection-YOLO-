# Dataset Status Report: Industrial Surface Defect Detection (NEU-DET)

**Date**: 2026-10-27  
**Workspace**: `c:\Users\arnav\OneDrive\Desktop\APPLIED AI 2`  
**Dataset Directory**: `raw_dataset/`  

---

## 1. Executive Summary

A comprehensive recursive inspection of the workspace was conducted. An authentic, complete industrial surface defect dataset was located under `raw_dataset/`. Detailed analysis establishes this dataset as the benchmark **NEU Surface Defect Database (NEU-DET)** from Northeastern University, containing hot-rolled steel strip surface defects.

All **1,800 images** and all **1,800 Pascal VOC XML annotations** are intact, verified, uncorrupted, and perfectly accounted for.

---

## 2. Directory Structure Discovered

The raw dataset is organized into `train` and `validation` subdirectories:

```
raw_dataset/
├── train/
│   ├── annotations/          # 1,439 XML files
│   └── images/               # 6 class subfolders (240 images each = 1,440 images)
│       ├── crazing/          # crazing_1.jpg to crazing_240.jpg
│       ├── inclusion/        # inclusion_1.jpg to inclusion_240.jpg
│       ├── patches/          # patches_1.jpg to patches_240.jpg
│       ├── pitted_surface/   # pitted_surface_1.jpg to pitted_surface_240.jpg
│       ├── rolled-in_scale/  # rolled-in_scale_1.jpg to rolled-in_scale_240.jpg
│       └── scratches/        # scratches_1.jpg to scratches_240.jpg
└── validation/
    ├── annotations/          # 361 XML files
    └── images/               # 6 class subfolders (60 images each = 360 images)
        ├── crazing/          # crazing_241.jpg to crazing_300.jpg
        ├── inclusion/        # inclusion_241.jpg to inclusion_300.jpg
        ├── patches/          # patches_241.jpg to patches_300.jpg
        ├── pitted_surface/   # pitted_surface_241.jpg to pitted_surface_300.jpg
        ├── rolled-in_scale/  # rolled-in_scale_241.jpg to rolled-in_scale_300.jpg
        └── scratches/        # scratches_241.jpg to scratches_300.jpg
```

---

## 3. Discovered Anomaly & Root Cause Analysis

During automated verification of file stems across images and annotations:
- **Raw Train**: 1,440 images, 1,439 annotations. Discrepancy: `crazing_240.jpg` had no matching XML in `train/annotations/`.
- **Raw Validation**: 360 images, 361 annotations. Discrepancy: `crazing_240.xml` was present in `validation/annotations/` while `crazing_240.jpg` was located in `train/images/crazing/`.
- **Global Check**:
  - Total unique images across workspace: **1,800**
  - Total unique annotations across workspace: **1,800**
  - Unmatched images: **0**
  - Unmatched annotations: **0**

**Decision**: 
The raw dataset directory `raw_dataset/` will be kept strictly read-only and unmodified to preserve provenance. In `src/prepare_dataset.py`, the indexer resolves annotations globally across `raw_dataset/**/annotations` by matching base stem names (`stem.xml` to `stem.jpg`), ensuring `crazing_240.jpg` is properly matched with its authentic annotation `crazing_240.xml` without data loss or corruption.

---

## 4. Image Properties & Verification

- **Format**: JPEG (`.jpg`)
- **Total Count**: 1,800 images
- **Dimensions**: 200 × 200 pixels (100% uniform across all 1,800 images)
- **Color Mode**: 3-channel RGB (grayscale steel strip surfaces saved as RGB)
- **Integrity**: Verified via PIL Image parser; **0 corrupted files**.

---

## 5. Defect Classes & Bounding Box Counts

Parsed from 1,800 Pascal VOC XML files:

| Class ID | Defect Class Name | Description | Bounding Box Instances | Images with Class |
| :---: | :--- | :--- | :---: | :---: |
| 0 | `crazing` | Surface hairline cracks and micro-fissures | 689 | 300 |
| 1 | `inclusion` | Non-metallic embedded particles/impurities | 1,011 | 300 |
| 2 | `patches` | Surface patch-like discoloration / oxidation | 881 | 300 |
| 3 | `pitted_surface` | Micro-cavities and pitting roughness | 432 | 300 |
| 4 | `rolled-in_scale` | Iron oxide pressed into surface during rolling | 628 | 300 |
| 5 | `scratches` | Longitudinal linear mechanical abrasions | 548 | 300 |
| **Total** | **6 classes** | **Industrial Steel Strip Defects** | **4,189** | **1,800** |

### Instance Complexity
- **Min boxes per image**: 1
- **Max boxes per image**: 9
- **Mean boxes per image**: 2.33
- **Images with 0 boxes**: 0 (all images contain at least one defect instance)

---

## 6. Dataset Splitting Plan

In accordance with project requirements and best practices:
1. **Preserve Training Pool**: The original 1,440 images (240 per class, 80%) form the `train` split.
2. **Held-Out Validation & Test Pool**: The original 360 held-out images (60 per class, 20%) are deterministically split 50/50 using a fixed random seed (`seed=42`) into:
   - `val` split: 180 images (30 per class, 10% of total)
   - `test` split: 180 images (30 per class, 10% of total)
3. **No Leakage**: Images from the same sequence or duplicates are never shared between splits. Splitting is stratified by defect category.

---

## 7. Status Sign-off

- [x] Workspace inspected and verified (`APPLIED AI 2`)
- [x] Raw dataset located (`raw_dataset/`)
- [x] All 1,800 images and 1,800 annotations accounted for
- [x] Zero fabricated annotations; zero deleted files
- [x] XML schema, coordinate boundaries, and class names validated
- [x] Conversion pipeline specifications ready for Phase 2
