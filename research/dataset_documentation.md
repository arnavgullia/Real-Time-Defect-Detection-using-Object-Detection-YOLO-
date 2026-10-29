# Dataset Documentation: NEU Surface Defect Database (NEU-DET)

**Course / Project**: Project-Based Learning (PBL-I) — Real-Time Industrial Defect Detection  
**Author / Team**: Applied AI Research Group  
**Date**: October 2026  
**Dataset Provenance**: Northeastern University (Shenyang, China), Surface Defect Database  

---

## 1. Dataset Selection and Justification

In modern hot-rolled steel strip manufacturing, surface quality inspection is a critical gatekeeping phase. Physical surface defects compromise structural integrity, tensile strength, and corrosion resistance. Manual visual inspection by human operators is error-prone, hazardous, and limited to low line speeds.

The **NEU Surface Defect Database (NEU-DET)** is the gold-standard benchmark in computer vision for metal surface quality control. It contains:
- Authentic industrial surface images collected from real rolling mills.
- Complex industrial background textures with varying illumination, oil stains, and roll wear.
- Six distinct defect classes capturing diverse geometrical and morphological distributions.

### Comparative Alternatives Evaluated

| Dataset Considered | Source | Primary Failure / Rejection Reason | Selected? |
| :--- | :--- | :--- | :---: |
| **NEU-DET** | Northeastern University | Ideal benchmark: authentic hot-rolled steel defects, uniform 200×200 resolution, diverse aspect ratios, multi-object images. | **Yes (Selected)** |
| **Severstal Steel Defect** | Kaggle / Severstal | Run-length encoded (RLE) segmentation masks, extreme class imbalance (mostly defect-free frames), high computational footprint for baseline training. | No (Rejected) |
| **KolektorSDD / SDD2** | Kolektor Group | Single-class binary crack defect detection on electrical commutators, lacks multi-class diversity. | No (Rejected) |
| **DAGM 2007** | DAGM / University of Heidelberg | Synthetic textured backgrounds with generated anomalies; does not represent genuine steel mill optical characteristics. | No (Rejected) |

---

## 2. Dataset Structure and Integrity

The raw dataset was audited under `raw_dataset/`:
- **Total Images**: 1,800
- **Total Pascal VOC XML Annotations**: 1,800
- **Native Resolution**: 200 × 200 pixels
- **Color Format**: 3-channel RGB (stored as grayscale representations across channels)
- **Zero Corrupted Files**: All images validated via PIL and OpenCV headers.

```
datasets/defect_dataset/
├── images/
│   ├── train/     # 1,440 images (80%)
│   ├── val/       #   180 images (10%)
│   └── test/      #   180 images (10%)
├── labels/
│   ├── train/     # 1,440 YOLO txt files (3,335 defect bounding boxes)
│   ├── val/       #   180 YOLO txt files (  441 defect bounding boxes)
│   └── test/      #   180 YOLO txt files (  413 defect bounding boxes)
├── class_mapping.json
└── data.yaml
```

---

## 3. Defect Taxonomy and Characteristics

| Class ID | Defect Category | Physical Description | Instance Count | Visual Signature |
| :---: | :--- | :--- | :---: | :--- |
| `0` | **crazing** | Network of fine surface micro-cracks | 689 | Hairline web-like fissures |
| `1` | **inclusion** | Non-metallic embedded impurities | 1,011 | Clustered dark spots / specks |
| `2` | **patches** | Surface oxidation and scale patches | 881 | Irregular dark/gray blotches |
| `3` | **pitted_surface** | Surface pitting and cavity depressions | 432 | Dense granular micro-depressions |
| `4` | **rolled-in_scale** | Iron oxide scale rolled into the steel | 628 | Fish-scale or streak indentations |
| `5` | **scratches** | Mechanical abrasion marks | 548 | High aspect ratio linear striations |
| **Total** | **6 Classes** | **Steel Strip Surface Defects** | **4,189** | **Mean: 2.33 boxes/image** |

---

## 4. Annotation Conversion and Normalization Protocol

Pascal VOC annotations define bounding boxes in absolute pixel coordinates `[xmin, ymin, xmax, ymax]`. YOLOv8 requires normalized coordinates `[class_id, x_center, y_center, width, height]`.

### Normalization Formula
Given image width $W=200$ and height $H=200$:
$$\text{width}_{\text{box}} = \text{clamp}(x_{\max}, 0, W) - \text{clamp}(x_{\min}, 0, W)$$
$$\text{height}_{\text{box}} = \text{clamp}(y_{\max}, 0, H) - \text{clamp}(y_{\min}, 0, H)$$
$$x_{\text{center}} = \frac{x_{\min} + \text{width}_{\text{box}} / 2}{W}$$
$$y_{\text{center}} = \frac{y_{\min} + \text{height}_{\text{box}} / 2}{H}$$
$$w = \frac{\text{width}_{\text{box}}}{W}, \quad h = \frac{\text{height}_{\text{box}}}{H}$$

All coordinates are rounded to 6 decimal places and verified to lie strictly in $[0.0, 1.0]$.
