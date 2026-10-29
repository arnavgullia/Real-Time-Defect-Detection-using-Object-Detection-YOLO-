"""
Dataset Preparation and XML-to-YOLO Annotation Converter for NEU-DET.
Converts Pascal VOC XML annotations to YOLOv8 format with stratified train/val/test splits.
"""

import os
import sys
import glob
import json
import shutil
import random
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.utils import (
    CLASS_NAMES,
    CLASS_TO_ID,
    setup_logger,
    voc_to_yolo_bbox,
    save_yaml,
)


def parse_voc_xml(xml_path: str, img_w: int, img_h: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Parses Pascal VOC XML annotation file.
    Returns list of valid bounding boxes and list of issues/warnings encountered.
    """
    boxes = []
    issues = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        issues.append(f"XML parse error in {xml_path}: {e}")
        return [], issues

    # Extract image size from XML if present, compare with actual
    size_elem = root.find("size")
    if size_elem is not None:
        try:
            xml_w = int(size_elem.find("width").text)
            xml_h = int(size_elem.find("height").text)
            if xml_w != img_w or xml_h != img_h:
                issues.append(
                    f"Dimension mismatch in {xml_path}: XML size ({xml_w}x{xml_h}) != actual image ({img_w}x{img_h})"
                )
        except Exception as e:
            issues.append(f"Could not read size tag in {xml_path}: {e}")

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            issues.append(f"Object missing 'name' in {xml_path}")
            continue

        raw_name = name_elem.text.strip().lower()
        if raw_name not in CLASS_TO_ID:
            issues.append(f"Unknown class '{raw_name}' in {xml_path}")
            continue

        class_id = CLASS_TO_ID[raw_name]
        bndbox = obj.find("bndbox")
        if bndbox is None:
            issues.append(f"Missing 'bndbox' element in {xml_path}")
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except Exception as e:
            issues.append(f"Invalid coordinate numbers in {xml_path}: {e}")
            continue

        # Check bounds
        clamped = False
        if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
            clamped = True

        try:
            xc, yc, w, h = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
            boxes.append({
                "class_name": raw_name,
                "class_id": class_id,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "x_center": xc,
                "y_center": yc,
                "width": w,
                "height": h,
                "clamped": clamped,
            })
            if clamped:
                issues.append(f"Clamped box [{xmin}, {ymin}, {xmax}, {ymax}] to [{img_w}x{img_h}] in {xml_path}")
        except Exception as e:
            issues.append(f"Failed to normalize box in {xml_path}: {e}")

    return boxes, issues


def prepare_dataset(
    raw_dir: str = "raw_dataset",
    output_dir: str = "datasets/defect_dataset",
    seed: int = 42,
    force_recreate: bool = True,
) -> Dict[str, Any]:
    """
    Converts NEU-DET raw dataset to YOLO format.
    Splits into train (1440), val (180), and test (180).
    """
    logger = setup_logger("PrepareDataset", "outputs/reports/prepare_dataset.log")
    logger.info("Starting dataset conversion and preparation...")

    raw_path = Path(raw_dir).resolve()
    dest_path = Path(output_dir).resolve()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw_path}")

    # Build index of all XML annotations across raw_dataset
    xml_files = list(raw_path.glob("**/*.xml"))
    xml_index: Dict[str, Path] = {}
    for xf in xml_files:
        stem = xf.stem.lower()
        if stem in xml_index:
            logger.warning(f"Duplicate XML stem found: {xf} vs {xml_index[stem]}")
        xml_index[stem] = xf
    logger.info(f"Indexed {len(xml_index)} unique XML annotations from raw dataset.")

    # Find all images in train and validation pools
    train_pool_imgs = []
    val_pool_imgs = []

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        train_pool_imgs.extend(list((raw_path / "train" / "images").glob(f"**/{ext}")))
        val_pool_imgs.extend(list((raw_path / "validation" / "images").glob(f"**/{ext}")))

    logger.info(f"Discovered {len(train_pool_imgs)} raw train images, {len(val_pool_imgs)} raw validation images.")

    # Sort deterministically
    train_pool_imgs = sorted(train_pool_imgs, key=lambda p: p.name)
    val_pool_imgs = sorted(val_pool_imgs, key=lambda p: p.name)

    # Class-aware split of val_pool (60 per class) into val (30) and test (30)
    rng = random.Random(seed)
    val_by_class: Dict[str, List[Path]] = {cls: [] for cls in CLASS_NAMES}
    for p in val_pool_imgs:
        cls_name = p.parent.name.lower()
        if cls_name in val_by_class:
            val_by_class[cls_name].append(p)
        else:
            # Fallback by stem prefix
            for c in CLASS_NAMES:
                if p.stem.lower().startswith(c):
                    val_by_class[c].append(p)
                    break

    split_val_imgs = []
    split_test_imgs = []

    for cls_name, imgs in val_by_class.items():
        sorted_cls_imgs = sorted(imgs, key=lambda p: p.name)
        rng.shuffle(sorted_cls_imgs)
        half = len(sorted_cls_imgs) // 2
        split_val_imgs.extend(sorted_cls_imgs[:half])
        split_test_imgs.extend(sorted_cls_imgs[half:])

    splits = {
        "train": train_pool_imgs,
        "val": sorted(split_val_imgs, key=lambda p: p.name),
        "test": sorted(split_test_imgs, key=lambda p: p.name),
    }

    logger.info(
        f"Assigned splits -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}"
    )

    # Prepare output directories
    if force_recreate and dest_path.exists():
        logger.info(f"Cleaning existing destination: {dest_path}")
        shutil.rmtree(dest_path)

    for split_name in ("train", "val", "test"):
        (dest_path / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (dest_path / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "total_images": 0,
        "total_boxes": 0,
        "class_counts": {c: 0 for c in CLASS_NAMES},
        "split_stats": {},
        "issues": [],
        "excluded_images": [],
    }

    # Process each split
    for split_name, img_paths in splits.items():
        split_box_count = 0
        split_class_counts = {c: 0 for c in CLASS_NAMES}
        split_img_count = 0

        for img_path in img_paths:
            stem = img_path.stem.lower()
            if stem not in xml_index:
                stats["excluded_images"].append({"image": str(img_path), "reason": "No matching XML found"})
                logger.error(f"Missing XML for image: {img_path}")
                continue

            xml_path = xml_index[stem]

            # Read actual image dimensions
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception as e:
                stats["excluded_images"].append({"image": str(img_path), "reason": f"Corrupt image file: {e}"})
                logger.error(f"Cannot read image {img_path}: {e}")
                continue

            # Parse XML
            boxes, issues = parse_voc_xml(str(xml_path), img_w, img_h)
            if issues:
                stats["issues"].extend(issues)

            # Copy image
            dest_img = dest_path / "images" / split_name / img_path.name
            shutil.copy2(img_path, dest_img)

            # Write YOLO label
            dest_label = dest_path / "labels" / split_name / f"{stem}.txt"
            with open(dest_label, "w", encoding="utf-8") as lf:
                for b in boxes:
                    lf.write(f"{b['class_id']} {b['x_center']:.6f} {b['y_center']:.6f} {b['width']:.6f} {b['height']:.6f}\n")
                    stats["class_counts"][b["class_name"]] += 1
                    split_class_counts[b["class_name"]] += 1
                    split_box_count += 1
                    stats["total_boxes"] += 1

            split_img_count += 1
            stats["total_images"] += 1

        stats["split_stats"][split_name] = {
            "image_count": split_img_count,
            "box_count": split_box_count,
            "class_distribution": split_class_counts,
        }

    # Save class mapping JSON
    class_mapping = {
        "class_to_id": CLASS_TO_ID,
        "id_to_class": {str(k): v for k, v in enumerate(CLASS_NAMES)},
        "classes": CLASS_NAMES,
    }
    with open(dest_path / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    # Write YOLO data.yaml
    # Use relative paths from data.yaml directory for portable training
    data_yaml_content = {
        "path": str(dest_path.as_posix()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    save_yaml(data_yaml_content, str(dest_path / "data.yaml"))

    # Also save conversion report
    reports_dir = WORKSPACE_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "conversion_report.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Generate Markdown Conversion Report
    md_report = [
        "# Dataset Conversion and Split Report (VOC to YOLOv8)",
        "",
        f"- **Total Processed Images**: {stats['total_images']}",
        f"- **Total Bounding Boxes**: {stats['total_boxes']}",
        f"- **Excluded Images**: {len(stats['excluded_images'])}",
        f"- **Reported Issues/Warnings**: {len(stats['issues'])}",
        "",
        "## Split Breakdown",
        "",
        "| Split | Images | Bounding Boxes | Avg Boxes/Image |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for s_name, s_data in stats["split_stats"].items():
        avg = s_data["box_count"] / max(1, s_data["image_count"])
        md_report.append(f"| `{s_name}` | {s_data['image_count']} | {s_data['box_count']} | {avg:.2f} |")

    md_report.extend([
        "",
        "## Class Distribution per Split",
        "",
        "| Class Name | Train Boxes | Val Boxes | Test Boxes | Total Boxes |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for cls in CLASS_NAMES:
        tb = stats["split_stats"]["train"]["class_distribution"].get(cls, 0)
        vb = stats["split_stats"]["val"]["class_distribution"].get(cls, 0)
        teb = stats["split_stats"]["test"]["class_distribution"].get(cls, 0)
        tot = stats["class_counts"].get(cls, 0)
        md_report.append(f"| `{cls}` | {tb} | {vb} | {teb} | {tot} |")

    with open(reports_dir / "conversion_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_report) + "\n")

    logger.info("Conversion complete! data.yaml and reports generated.")
    return stats


if __name__ == "__main__":
    prepare_dataset()
