# Exploratory Data Analysis Report: NEU-DET Surface Defect Dataset

## 1. Executive Summary
- **Total Dataset Images**: 1800
- **Total Defect Instances**: 4189
- **Uniform Image Resolution**: 200 × 200 pixels (3 channels)
- **Defect Classes**: 6 categories (`crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches`)
- **Average Defect Density**: 2.33 defects per image (range: 1 to 9)

## 2. Dataset Split Breakdown

| Split | Images | Defect Instances | Avg Defects/Image | Percentage of Dataset |
| :--- | :---: | :---: | :---: | :---: |
| `test` | 180 | 413 | 2.29 | 10.0% |
| `train` | 1440 | 3335 | 2.32 | 80.0% |
| `val` | 180 | 441 | 2.45 | 10.0% |

## 3. Class Imbalance and Instance Distribution

| Defect Class | Images | Instances | Ratio to Max | % of Instances |
| :--- | :---: | :---: | :---: | :---: |
| `crazing` | 300 | 689 | 1 : 1.47 | 16.45% |
| `inclusion` | 382 | 1011 | 1 : 1.00 | 24.13% |
| `patches` | 342 | 881 | 1 : 1.15 | 21.03% |
| `pitted_surface` | 301 | 432 | 1 : 2.34 | 10.31% |
| `rolled-in_scale` | 300 | 628 | 1 : 1.61 | 14.99% |
| `scratches` | 300 | 548 | 1 : 1.84 | 13.08% |

## 4. Defect Scale Analysis (COCO Metric)
- **Small Defect**: Area < 32² px = 1,024 px² (Normalized area < 0.0256)
- **Medium Defect**: 32² px ≤ Area ≤ 96² px (1,024 to 9,216 px², Normalized 0.0256 to 0.2304)
- **Large Defect**: Area > 96² px = 9,216 px² (Normalized area > 0.2304)

### Scale Breakdown by Category

| Defect Class | Large (>96² px) | Medium (32²-96² px) | Small (<32² px) | Total |
| :--- | :---: | :---: | :---: | :---: |
| `crazing` | 294 | 395 | 0 | 689 |
| `inclusion` | 36 | 646 | 329 | 1011 |
| `patches` | 141 | 684 | 56 | 881 |
| `pitted_surface` | 374 | 55 | 3 | 432 |
| `rolled-in_scale` | 110 | 502 | 16 | 628 |
| `scratches` | 12 | 493 | 43 | 548 |

## 5. Architectural and Experimental Insights

### A. Model Architecture Selection
The NEU-DET defects range from tiny micro-pits (`pitted_surface` and `inclusion`) to extensive horizontal/vertical linear marks (`scratches`) and sprawling surface degradation (`patches`, `crazing`).
- **Anchor-Free YOLOv8**: Traditional anchor-based detectors (like standard YOLOv5 without anchor re-clustering) suffer from high IoU sensitivity on extreme aspect ratios. YOLOv8's anchor-free TaskAlignedAssigner directly regresses box bounds, making it far superior for varied defect geometries.
- **YOLOv8n Baseline**: With ~3.2M parameters, YOLOv8n achieves high FPS on industrial edge devices while avoiding severe overfitting on 1,440 training images.

### B. Input Image Resolution Consideration
Native images are 200 × 200 px.
- Downsampling (e.g. to 160 px) would cause severe feature collapse on micro-inclusions.
- Training at native or standard YOLO resolution (e.g., 320 px or 640 px) introduces interpolation upsampling. 640 px gives finer grid resolution in the P3/8 feature map, critical for detecting small defects.

### C. Augmentation Strategy
- **Preserve Linear Artifacts**: Geometric distortion (heavy shear or extreme perspective) can distort longitudinal scratches. Flips (horizontal & vertical) are safe because rolled steel can travel in either direction.
- **HSV Augmentation**: Mild hue/saturation/value variation helps adapt to lighting changes in hot-rolling mill inspection chambers.
- **Mosaic Augmentation**: Effective for defect density, but small defect bounding boxes must not be cut off.
