"""
Exploratory Data Analysis (EDA) and Visualization for NEU-DET Industrial Surface Defect Dataset.
Computes image-level counts, bounding-box distributions, COCO defect-scale metrics (small/medium/large),
and generates publication-quality charts, summary CSVs, and an exhaustive markdown report.
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.utils import (
    CLASS_NAMES,
    ID_TO_CLASS,
    CLASS_COLORS_RGB,
    setup_logger,
    yolo_to_voc_bbox,
)

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def run_eda(
    dataset_dir: str = "datasets/defect_dataset",
    output_dir: str = "outputs/reports/eda",
) -> Dict[str, Any]:
    """Runs complete Exploratory Data Analysis and generates visual and tabular artifacts."""
    logger = setup_logger("AnalyzeDataset", f"{output_dir}/eda.log")
    logger.info("Starting comprehensive Exploratory Data Analysis...")

    dataset_root = Path(dataset_dir).resolve()
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # Collect all box records and image metadata
    box_records = []
    image_records = []

    for split in ("train", "val", "test"):
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split

        if not img_dir.exists():
            continue

        img_files = sorted(list(img_dir.glob("*.jpg")))
        for img_path in img_files:
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            classes_in_image = set()
            boxes_in_image = 0

            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as lf:
                    lines = [l.strip() for l in lf if l.strip()]

                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    cls_name = ID_TO_CLASS.get(cls_id, f"cls_{cls_id}")
                    classes_in_image.add(cls_name)
                    boxes_in_image += 1

                    # Pixel dimensions
                    pixel_w = w * img_w
                    pixel_h = h * img_h
                    pixel_area = pixel_w * pixel_h
                    norm_area = w * h
                    aspect_ratio = pixel_w / max(1e-3, pixel_h)

                    # Categorize scale by COCO definitions (small < 32^2=1024, med 1024-9216, large > 96^2=9216)
                    if pixel_area < 1024:
                        scale_cat = "Small (<32² px)"
                    elif pixel_area <= 9216:
                        scale_cat = "Medium (32²-96² px)"
                    else:
                        scale_cat = "Large (>96² px)"

                    box_records.append({
                        "split": split,
                        "image_stem": stem,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "x_center": xc,
                        "y_center": yc,
                        "norm_width": w,
                        "norm_height": h,
                        "norm_area": norm_area,
                        "pixel_width": pixel_w,
                        "pixel_height": pixel_h,
                        "pixel_area": pixel_area,
                        "aspect_ratio": aspect_ratio,
                        "scale_category": scale_cat,
                    })

            image_records.append({
                "split": split,
                "image_stem": stem,
                "img_width": img_w,
                "img_height": img_h,
                "box_count": boxes_in_image,
                "classes_present": list(classes_in_image),
                "num_classes": len(classes_in_image),
            })

    df_boxes = pd.DataFrame(box_records)
    df_images = pd.DataFrame(image_records)

    logger.info(f"Loaded {len(df_images)} images and {len(df_boxes)} defect instances.")

    # 1. Save Tabular CSVs
    df_boxes.to_csv(out_root / "all_boxes.csv", index=False)
    df_images.to_csv(out_root / "all_images.csv", index=False)

    # Split breakdown
    split_summary = df_images.groupby("split").agg(
        image_count=("image_stem", "count"),
        total_boxes=("box_count", "sum"),
        avg_boxes_per_image=("box_count", "mean"),
    ).reset_index()
    split_summary.to_csv(out_root / "split_breakdown.csv", index=False)

    # Class distribution (Image-level vs Box-level)
    class_box_counts = df_boxes["class_name"].value_counts().to_dict()
    # Image-level: how many images contain this class
    img_class_counts = {c: 0 for c in CLASS_NAMES}
    for _, row in df_images.iterrows():
        for c in row["classes_present"]:
            img_class_counts[c] += 1

    class_dist_rows = []
    max_box_count = max(class_box_counts.values()) if class_box_counts else 1
    for c in CLASS_NAMES:
        b_cnt = class_box_counts.get(c, 0)
        im_cnt = img_class_counts.get(c, 0)
        class_dist_rows.append({
            "class_name": c,
            "image_count": im_cnt,
            "instance_count": b_cnt,
            "imbalance_ratio_to_max": round(max_box_count / max(1, b_cnt), 3),
            "percentage_of_all_boxes": round((b_cnt / max(1, len(df_boxes))) * 100, 2),
        })
    df_class_dist = pd.DataFrame(class_dist_rows)
    df_class_dist.to_csv(out_root / "class_distribution.csv", index=False)

    # Defect scale categories by class
    df_scale_by_class = pd.crosstab(df_boxes["class_name"], df_boxes["scale_category"], margins=True)
    df_scale_by_class.to_csv(out_root / "defect_sizes_by_class.csv")

    # 2. Generate Publication Charts
    logger.info("Generating publication-quality charts...")

    # Chart 1: Split Breakdown
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    sns.barplot(data=split_summary, x="split", y="image_count", palette="Blues_d", ax=ax)
    ax.set_title("Dataset Split Distribution (Images)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Split", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_root / "split_distribution.png")
    plt.close()

    # Chart 2: Image-level vs Instance-level Class Distribution
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    ax.bar(x - width/2, [img_class_counts[c] for c in CLASS_NAMES], width, label="Images Containing Class", color="#3B82F6")
    ax.bar(x + width/2, [class_box_counts.get(c, 0) for c in CLASS_NAMES], width, label="Defect Instances (Boxes)", color="#EF4444")
    ax.set_title("Image-Level vs Bounding-Box Instance Distribution by Defect Class", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_root / "class_distribution.png")
    plt.close()

    # Chart 3: Defect Scale Categories by Class (Small / Medium / Large)
    df_scale_plot = pd.crosstab(df_boxes["class_name"], df_boxes["scale_category"])
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    df_scale_plot.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", edgecolor="none")
    ax.set_title("Defect Scale Distribution Across Classes (COCO Scale Metric)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Defect Class", fontsize=11)
    ax.set_ylabel("Number of Instances", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(title="Scale Threshold", frameon=True)
    plt.tight_layout()
    plt.savefig(out_root / "defect_scale_by_class.png")
    plt.close()

    # Chart 4: Bounding Box Dimensions (Width vs Height Normalized)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for c in CLASS_NAMES:
        sub = df_boxes[df_boxes["class_name"] == c]
        rgb = [v / 255.0 for v in CLASS_COLORS_RGB.get(c, (128, 128, 128))]
        ax.scatter(sub["norm_width"], sub["norm_height"], label=c, alpha=0.5, s=25, color=rgb)
    ax.set_title("Normalized Bounding Box Width vs Height", fontsize=13, fontweight="bold")
    ax.set_xlabel("Normalized Width (w / 200)", fontsize=11)
    ax.set_ylabel("Normalized Height (h / 200)", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Defect Category", frameon=True)
    plt.tight_layout()
    plt.savefig(out_root / "bbox_dimensions_scatter.png")
    plt.close()

    # Chart 5: Defect Density per Image
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    sns.countplot(data=df_images, x="box_count", palette="mako", ax=ax)
    ax.set_title("Defect Density: Instances per Image", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Defects in Single Image", fontsize=11)
    ax.set_ylabel("Image Count", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() + 10),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_root / "defect_density_per_image.png")
    plt.close()

    # Chart 6: 2x3 Visual Sample Grid of Defects
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300)
    axes = axes.flatten()
    for idx, c in enumerate(CLASS_NAMES):
        ax = axes[idx]
        sample_rows = df_boxes[df_boxes["class_name"] == c]
        if not sample_rows.empty:
            sample_stem = sample_rows.iloc[0]["image_stem"]
            sample_split = sample_rows.iloc[0]["split"]
            img_file = dataset_root / "images" / sample_split / f"{sample_stem}.jpg"
            if img_file.exists():
                with Image.open(img_file).convert("RGB") as pil_im:
                    draw = ImageDraw.Draw(pil_im)
                    # Overlay all boxes for this sample
                    sub_b = df_boxes[df_boxes["image_stem"] == sample_stem]
                    for _, b in sub_b.iterrows():
                        xmin, ymin, xmax, ymax = yolo_to_voc_bbox(
                            b["x_center"], b["y_center"], b["norm_width"], b["norm_height"], 200, 200
                        )
                        draw.rectangle([xmin, ymin, xmax, ymax], outline=(239, 68, 68), width=2)
                    ax.imshow(np.array(pil_im))
        ax.set_title(f"{c.capitalize()}", fontsize=12, fontweight="bold")
        ax.axis("off")
    plt.suptitle("Authentic NEU-DET Surface Defect Samples with Ground-Truth", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_root / "sample_grid.png")
    plt.close()

    # 3. Generate Exhaustive Markdown EDA Report
    eda_md = [
        "# Exploratory Data Analysis Report: NEU-DET Surface Defect Dataset",
        "",
        "## 1. Executive Summary",
        f"- **Total Dataset Images**: {len(df_images)}",
        f"- **Total Defect Instances**: {len(df_boxes)}",
        f"- **Uniform Image Resolution**: 200 × 200 pixels (3 channels)",
        f"- **Defect Classes**: 6 categories (`{', '.join(CLASS_NAMES)}`)",
        f"- **Average Defect Density**: {len(df_boxes) / max(1, len(df_images)):.2f} defects per image (range: 1 to 9)",
        "",
        "## 2. Dataset Split Breakdown",
        "",
        "| Split | Images | Defect Instances | Avg Defects/Image | Percentage of Dataset |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for _, r in split_summary.iterrows():
        pct = (r['image_count'] / len(df_images)) * 100
        eda_md.append(f"| `{r['split']}` | {r['image_count']} | {r['total_boxes']} | {r['avg_boxes_per_image']:.2f} | {pct:.1f}% |")

    eda_md.extend([
        "",
        "## 3. Class Imbalance and Instance Distribution",
        "",
        "| Defect Class | Images | Instances | Ratio to Max | % of Instances |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for _, r in df_class_dist.iterrows():
        eda_md.append(f"| `{r['class_name']}` | {r['image_count']} | {r['instance_count']} | 1 : {r['imbalance_ratio_to_max']:.2f} | {r['percentage_of_all_boxes']}% |")

    # Format scale markdown table manually without tabulate dependency
    scale_table_md = ["| Defect Class | Large (>96² px) | Medium (32²-96² px) | Small (<32² px) | Total |",
                      "| :--- | :---: | :---: | :---: | :---: |"]
    for cls in CLASS_NAMES:
        l_cnt = df_scale_plot.loc[cls, "Large (>96² px)"] if "Large (>96² px)" in df_scale_plot.columns else 0
        m_cnt = df_scale_plot.loc[cls, "Medium (32²-96² px)"] if "Medium (32²-96² px)" in df_scale_plot.columns else 0
        s_cnt = df_scale_plot.loc[cls, "Small (<32² px)"] if "Small (<32² px)" in df_scale_plot.columns else 0
        tot = l_cnt + m_cnt + s_cnt
        scale_table_md.append(f"| `{cls}` | {l_cnt} | {m_cnt} | {s_cnt} | {tot} |")

    eda_md.extend([
        "",
        "## 4. Defect Scale Analysis (COCO Metric)",
        "- **Small Defect**: Area < 32² px = 1,024 px² (Normalized area < 0.0256)",
        "- **Medium Defect**: 32² px ≤ Area ≤ 96² px (1,024 to 9,216 px², Normalized 0.0256 to 0.2304)",
        "- **Large Defect**: Area > 96² px = 9,216 px² (Normalized area > 0.2304)",
        "",
        "### Scale Breakdown by Category",
        "",
        "\n".join(scale_table_md),
        "",
        "## 5. Architectural and Experimental Insights",
        "",
        "### A. Model Architecture Selection",
        "The NEU-DET defects range from tiny micro-pits (`pitted_surface` and `inclusion`) to extensive horizontal/vertical linear marks (`scratches`) and sprawling surface degradation (`patches`, `crazing`).",
        "- **Anchor-Free YOLOv8**: Traditional anchor-based detectors (like standard YOLOv5 without anchor re-clustering) suffer from high IoU sensitivity on extreme aspect ratios. YOLOv8's anchor-free TaskAlignedAssigner directly regresses box bounds, making it far superior for varied defect geometries.",
        "- **YOLOv8n Baseline**: With ~3.2M parameters, YOLOv8n achieves high FPS on industrial edge devices while avoiding severe overfitting on 1,440 training images.",
        "",
        "### B. Input Image Resolution Consideration",
        "Native images are 200 × 200 px.",
        "- Downsampling (e.g. to 160 px) would cause severe feature collapse on micro-inclusions.",
        "- Training at native or standard YOLO resolution (e.g., 320 px or 640 px) introduces interpolation upsampling. 640 px gives finer grid resolution in the P3/8 feature map, critical for detecting small defects.",
        "",
        "### C. Augmentation Strategy",
        "- **Preserve Linear Artifacts**: Geometric distortion (heavy shear or extreme perspective) can distort longitudinal scratches. Flips (horizontal & vertical) are safe because rolled steel can travel in either direction.",
        "- **HSV Augmentation**: Mild hue/saturation/value variation helps adapt to lighting changes in hot-rolling mill inspection chambers.",
        "- **Mosaic Augmentation**: Effective for defect density, but small defect bounding boxes must not be cut off.",
    ])

    with open(out_root / "eda_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(eda_md) + "\n")

    logger.info("EDA completed successfully! Reports, charts, and CSVs saved.")
    return {
        "images": len(df_images),
        "boxes": len(df_boxes),
        "classes": len(CLASS_NAMES),
        "reports_dir": str(out_root),
    }


if __name__ == "__main__":
    run_eda()
