# Research Plan & Experiment Matrix: Industrial Defect Detection

**Project**: Real-Time Industrial Defect Detection Using YOLOv8  
**Dataset**: NEU-DET (1,800 images: 1,440 train, 180 val, 180 test)  
**Hardware Profile**: AMD Ryzen 7 7435HS (Host CPU Mode), Extensible to CUDA GPU  

---

## 1. Experimental Methodology & Variable Control

To establish scientific rigor in accordance with Project-Based Learning review standards, all comparative experiments follow a strict single-variable perturbation protocol:
- **Consistent Splits**: The deterministic split (train: 1,440, val: 180, test: 180, `seed=42`) is frozen across all experiments.
- **Evaluation Protocol**: Models are evaluated on the identical held-out validation set during development; the test set is reserved exclusively for the final selected model checkpoint to prevent hyperparameter overfitting.
- **Controlled Baselines**: All perturbations are benchmarked directly against the YOLOv8n baseline (`EXP-01`).

---

## 2. Controlled Experiment Matrix

### Study A: Input Image Resolution Impact
- **Hypothesis**: The native dataset resolution is 200×200. Upscaling to 640×640 or 960×960 increases the spatial sampling grid in P3/8 feature maps, improving localization of micro-inclusions and pitting, but increases latency proportionally.
- **Independent Variable**: Image dimension ($S \in \{320, 640, 960\}$).
- **Control Variables**: Batch size = 16 (8 for 960), Optimizer = AdamW, Epochs = 40, Learning Rate = 0.01.

| Exp ID | Resolution | Primary Target | Expected Trade-off |
| :---: | :---: | :--- | :--- |
| `EXP-02` | **320 × 320** | High FPS edge deployment | Lower mAP on micro-inclusions |
| `EXP-03` | **640 × 640** | Standard industrial baseline | Balanced mAP and real-time FPS |
| `EXP-04` | **960 × 960** | Micro-fissure & craze detection | Higher memory & inference latency |

---

### Study B: Domain-Specific Defect-Preserving Augmentation
- **Hypothesis**: Standard vision augmentations (heavy shear, extreme random cropping) can sever thin longitudinal scratches or distort micro-cracks. Symmetrical vertical/horizontal flips preserve rolled steel defect physics, since rolled steel travel direction is bidirectional in reverse-rolling mills.
- **Independent Variable**: Augmentation pipeline.
- **Control Variables**: Architecture = YOLOv8n, Resolution = 640px, Epochs = 40.

| Exp ID | Augmentation Strategy | Configuration Parameters |
| :---: | :--- | :--- |
| `EXP-05` | **Minimal Augmentation** | `mosaic=0.0`, `mixup=0.0`, `fliplr=0.0`, `flipud=0.0` |
| `EXP-06` | **Defect-Preserving Strategy** | `fliplr=0.5`, `flipud=0.5`, `hsv_h=0.015`, `hsv_s=0.4`, `hsv_v=0.3`, `mosaic=0.5`, `close_mosaic=10` |

---

### Study C: Model Capacity Scaling (Backbone Depth)
- **Hypothesis**: YOLOv8s (~11.2M parameters) provides richer multi-scale receptive fields than YOLOv8n (~3.0M parameters), improving separation of complex overlapping defects (e.g. crazing on oxidized patches).
- **Independent Variable**: Model scale (`yolov8n` vs `yolov8s`).
- **Control Variables**: Resolution = 640px, Batch = 8, Epochs = 40.

---

### Study D: Anchor-Box Research Study (Anchor-Free vs Anchor-Based Benchmark)

> [!IMPORTANT]
> **Architectural Note on YOLOv8 Anchor-Free Design**:  
> YOLOv8 is strictly an **anchor-free** detector that utilizes a decoupled head with the **Task-Aligned Assigner (TAL)**. It directly regresses continuous bounding box offsets from feature grid centers ($x, y, w, h$) without predefined anchor box cluster priors. Traditional anchor clustering (e.g., k-means anchor tuning) is structurally not applicable inside YOLOv8.

To scientifically address the anchor box research requirement:
- We deploy a parallel benchmark using **YOLOv5n** (anchor-based with k-means autoanchor clustering).
- **Baseline Protocol**: YOLOv5n with standard COCO default anchor priors:
  `[(10,13, 16,30, 33,23), (30,61, 62,45, 59,119), (116,90, 156,198, 373,326)]`.
- **Experimental Protocol**: YOLOv5n with dataset-fitted anchors derived via **k-means clustering + genetic algorithm (GA)** directly on NEU-DET bounding box aspect ratios.
- **Comparison Metrics**: Precision, Recall, mAP@50, mAP@50-95, BBox recall rate (BPR), Model Size (MB), FPS.

---

## 3. Small-Defect Specific Evaluation Metric

To isolate performance on safety-critical micro-defects:
$$\text{AP}_{\text{small}} = \text{AP for bounding box area } < 1024 \text{ px}^2 \text{ (COCO standard)}$$
Special attention is directed toward `inclusion` and `pitted_surface` categories, which comprise >70% of all small defects in NEU-DET.
