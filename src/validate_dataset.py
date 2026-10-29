"""
Dataset Validation and Integrity Verification for YOLOv8 NEU-DET Dataset.
Verifies structure, coordinates, label formats, image integrity, and exports visual verification samples.
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image, ImageDraw, ImageFont

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
    load_yaml,
)


def validate_split(
    dataset_root: Path,
    split_name: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Validates images and labels within a specific split."""
    images_dir = dataset_root / "images" / split_name
    labels_dir = dataset_root / "labels" / split_name

    split_report: Dict[str, Any] = {
        "split": split_name,
        "image_count": 0,
        "label_count": 0,
        "total_boxes": 0,
        "class_counts": {c: 0 for c in CLASS_NAMES},
        "errors": [],
        "warnings": [],
        "passed": True,
    }

    if not images_dir.exists() or not labels_dir.exists():
        split_report["errors"].append(f"Directory missing for split '{split_name}'")
        split_report["passed"] = False
        return split_report

    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    label_files = sorted([f for f in labels_dir.iterdir() if f.suffix.lower() == ".txt"])

    split_report["image_count"] = len(image_files)
    split_report["label_count"] = len(label_files)

    image_stems = {f.stem.lower(): f for f in image_files}
    label_stems = {f.stem.lower(): f for f in label_files}

    # Check for missing pairings
    missing_labels = set(image_stems.keys()) - set(label_stems.keys())
    missing_images = set(label_stems.keys()) - set(image_stems.keys())

    if missing_labels:
        split_report["errors"].append(f"Images without labels: {len(missing_labels)} (e.g. {list(missing_labels)[:3]})")
        split_report["passed"] = False

    if missing_images:
        split_report["errors"].append(f"Labels without images: {len(missing_images)} (e.g. {list(missing_images)[:3]})")
        split_report["passed"] = False

    # Validate each image and label pair
    for stem, img_path in image_stems.items():
        # Check image readability
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                if img_w <= 0 or img_h <= 0:
                    split_report["errors"].append(f"Invalid dimensions {img_w}x{img_h} in {img_path.name}")
                    split_report["passed"] = False
        except Exception as e:
            split_report["errors"].append(f"Cannot read image {img_path.name}: {e}")
            split_report["passed"] = False
            continue

        lbl_path = label_stems.get(stem)
        if not lbl_path or not lbl_path.exists():
            continue

        # Validate label lines
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line_idx, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    split_report["errors"].append(
                        f"Line {line_idx} in {lbl_path.name} has {len(parts)} fields instead of 5: '{line}'"
                    )
                    split_report["passed"] = False
                    continue

                try:
                    cls_id = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError as ve:
                    split_report["errors"].append(f"Non-numeric field in {lbl_path.name}:{line_idx}: {ve}")
                    split_report["passed"] = False
                    continue

                if cls_id not in ID_TO_CLASS:
                    split_report["errors"].append(f"Invalid class ID {cls_id} in {lbl_path.name}:{line_idx}")
                    split_report["passed"] = False
                    continue

                cls_name = ID_TO_CLASS[cls_id]

                # Check YOLO normalized bounds
                if not (0.0 <= xc <= 1.0) or not (0.0 <= yc <= 1.0):
                    split_report["errors"].append(
                        f"Box center ({xc}, {yc}) outside [0, 1] in {lbl_path.name}:{line_idx}"
                    )
                    split_report["passed"] = False

                if w <= 0.0 or h <= 0.0:
                    split_report["errors"].append(
                        f"Non-positive box width/height ({w}, {h}) in {lbl_path.name}:{line_idx}"
                    )
                    split_report["passed"] = False

                if (xc - w / 2.0 < -1e-4) or (xc + w / 2.0 > 1.0001) or (yc - h / 2.0 < -1e-4) or (yc + h / 2.0 > 1.0001):
                    split_report["warnings"].append(
                        f"Box bounds slightly exceed [0, 1] in {lbl_path.name}:{line_idx}"
                    )

                split_report["class_counts"][cls_name] += 1
                split_report["total_boxes"] += 1

        except Exception as e:
            split_report["errors"].append(f"Failed to read label {lbl_path.name}: {e}")
            split_report["passed"] = False

    return split_report


def generate_visual_checks(
    dataset_root: Path,
    output_dir: Path,
    num_samples_per_split: int = 5,
    seed: int = 42,
) -> List[str]:
    """Draws ground truth bounding boxes on sample images and saves them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    saved_images = []

    for split in ("train", "val", "test"):
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split

        if not img_dir.exists():
            continue

        images = sorted(list(img_dir.glob("*.jpg")))
        if not images:
            continue

        samples = rng.sample(images, min(num_samples_per_split, len(images)))

        for img_path in samples:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            with Image.open(img_path).convert("RGB") as im:
                img_w, img_h = im.size
                draw = ImageDraw.Draw(im)

                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]

                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    cls_name = ID_TO_CLASS.get(cls_id, f"cls_{cls_id}")
                    color = CLASS_COLORS_RGB.get(cls_name, (255, 0, 0))

                    xmin, ymin, xmax, ymax = yolo_to_voc_bbox(xc, yc, w, h, img_w, img_h)

                    # Draw rectangle with thickness
                    for offset in range(2):
                        draw.rectangle(
                            [xmin - offset, ymin - offset, xmax + offset, ymax + offset],
                            outline=color,
                        )

                    # Draw text banner
                    label_text = f"{cls_name} ({cls_id})"
                    text_bbox = draw.textbbox((xmin, max(0, ymin - 14)), label_text)
                    draw.rectangle(text_bbox, fill=color)
                    draw.text((xmin + 2, max(0, ymin - 14)), label_text, fill=(255, 255, 255))

                save_path = output_dir / f"check_{split}_{img_path.name}"
                im.save(save_path)
                saved_images.append(str(save_path))

    return saved_images


def validate_dataset(
    dataset_dir: str = "datasets/defect_dataset",
    output_reports_dir: str = "outputs/reports",
) -> bool:
    """Runs complete dataset validation and outputs JSON and Markdown reports."""
    logger = setup_logger("ValidateDataset", f"{output_reports_dir}/dataset_validation.log")
    logger.info("Starting comprehensive dataset validation...")

    dataset_root = Path(dataset_dir).resolve()
    reports_root = Path(output_reports_dir).resolve()
    reports_root.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        logger.error(f"Dataset path does not exist: {dataset_root}")
        return False

    # Check data.yaml
    data_yaml_path = dataset_root / "data.yaml"
    yaml_valid = False
    yaml_info = {}
    if data_yaml_path.exists():
        try:
            yaml_info = load_yaml(str(data_yaml_path))
            if "names" in yaml_info and len(yaml_info["names"]) == len(CLASS_NAMES):
                yaml_valid = True
                logger.info(f"Verified data.yaml: {len(yaml_info['names'])} classes configured.")
        except Exception as e:
            logger.error(f"Corrupt data.yaml: {e}")

    overall_report: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "data_yaml_valid": yaml_valid,
        "splits": {},
        "total_images": 0,
        "total_boxes": 0,
        "overall_passed": True,
        "visual_checks_generated": [],
    }

    for split in ("train", "val", "test"):
        split_res = validate_split(dataset_root, split, logger)
        overall_report["splits"][split] = split_res
        overall_report["total_images"] += split_res["image_count"]
        overall_report["total_boxes"] += split_res["total_boxes"]
        if not split_res["passed"]:
            overall_report["overall_passed"] = False
            logger.error(f"Split '{split}' FAILED validation with {len(split_res['errors'])} errors!")
        else:
            logger.info(
                f"Split '{split}' PASSED: {split_res['image_count']} images, {split_res['total_boxes']} boxes."
            )

    # Visual checks
    visual_dir = reports_root / "annotation_checks"
    visual_samples = generate_visual_checks(dataset_root, visual_dir, num_samples_per_split=6)
    overall_report["visual_checks_generated"] = visual_samples
    logger.info(f"Generated {len(visual_samples)} visual annotation verification samples in {visual_dir}")

    # Save JSON report
    with open(reports_root / "dataset_validation.json", "w", encoding="utf-8") as f:
        json.dump(overall_report, f, indent=2)

    # Save Markdown report
    md_lines = [
        "# Dataset Validation Report",
        "",
        f"- **Status**: {'PASSED (Ready for Baseline Training)' if overall_report['overall_passed'] else 'FAILED'}",
        f"- **Dataset Path**: `{dataset_root}`",
        f"- **data.yaml Verified**: {'Yes' if yaml_valid else 'No'}",
        f"- **Total Validated Images**: {overall_report['total_images']}",
        f"- **Total Validated Bounding Boxes**: {overall_report['total_boxes']}",
        "",
        "## Split Verification Summary",
        "",
        "| Split | Images | Labels | Bounding Boxes | Errors | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s_name, s_data in overall_report["splits"].items():
        status_badge = "PASSED" if s_data["passed"] else "FAILED"
        md_lines.append(
            f"| `{s_name}` | {s_data['image_count']} | {s_data['label_count']} | {s_data['total_boxes']} | {len(s_data['errors'])} | {status_badge} |"
        )

    md_lines.extend([
        "",
        "## Visual Annotation Checks",
        f"Generated {len(visual_samples)} bounding box overlays stored under `outputs/reports/annotation_checks/`.",
    ])

    with open(reports_root / "dataset_validation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return overall_report["overall_passed"]


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
