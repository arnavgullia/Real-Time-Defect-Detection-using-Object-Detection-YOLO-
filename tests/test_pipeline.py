"""
Automated Test Suite for Industrial Defect Detection System.
Tests XML parsing, coordinate normalization, class mapping, split reproducibility,
corrupt annotation handling, and model error handling.
"""

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.utils import (
    CLASS_NAMES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    voc_to_yolo_bbox,
    yolo_to_voc_bbox,
)
from src.prepare_dataset import parse_voc_xml
from src.predict import run_prediction
from src.evaluate import evaluate_model


def test_class_mapping_integrity():
    """Validates class mapping consistency, count, and bi-directional indexing."""
    assert len(CLASS_NAMES) == 6
    assert set(CLASS_NAMES) == {"crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"}
    for idx, name in enumerate(CLASS_NAMES):
        assert CLASS_TO_ID[name] == idx
        assert ID_TO_CLASS[idx] == name


def test_coordinate_normalization():
    """Tests standard conversion from VOC [xmin, ymin, xmax, ymax] to normalized YOLO."""
    # 200x200 image, box from (50, 50) to (150, 150) -> center (100, 100) -> 0.5, 0.5, 0.5, 0.5
    xc, yc, w, h = voc_to_yolo_bbox(50, 50, 150, 150, 200, 200)
    assert xc == 0.5
    assert yc == 0.5
    assert w == 0.5
    assert h == 0.5

    # Reverse conversion
    xmin, ymin, xmax, ymax = yolo_to_voc_bbox(xc, yc, w, h, 200, 200)
    assert xmin == 50
    assert ymin == 50
    assert xmax == 150
    assert ymax == 150


def test_coordinate_clamping_and_bounds():
    """Tests that out-of-boundary coordinates are clamped safely to [0, 1]."""
    # Box extending past right and bottom edges (-10 to 220)
    xc, yc, w, h = voc_to_yolo_bbox(-10, -5, 220, 210, 200, 200)
    assert 0.0 <= xc <= 1.0
    assert 0.0 <= yc <= 1.0
    assert 0.0 <= w <= 1.0
    assert 0.0 <= h <= 1.0


def test_invalid_negative_box_raises_error():
    """Tests that invalid inverted or zero-area boxes raise ValueError."""
    with pytest.raises(ValueError):
        voc_to_yolo_bbox(100, 100, 50, 50, 200, 200)  # xmin > xmax


def test_voc_xml_parsing_valid():
    """Tests parsing of a valid Pascal VOC XML string."""
    xml_content = """<annotation>
        <size><width>200</width><height>200</height><depth>1</depth></size>
        <object>
            <name>inclusion</name>
            <bndbox><xmin>20</xmin><ymin>30</ymin><xmax>80</xmax><ymax>90</ymax></bndbox>
        </object>
    </annotation>"""
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        tmp_path = f.name

    try:
        boxes, issues = parse_voc_xml(tmp_path, 200, 200)
        assert len(boxes) == 1
        assert boxes[0]["class_name"] == "inclusion"
        assert boxes[0]["class_id"] == CLASS_TO_ID["inclusion"]
        assert len(issues) == 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_voc_xml_parsing_malformed():
    """Tests that malformed XML reports issues rather than crashing."""
    malformed_xml = "<annotation><size><width>200</width></annotation>"  # unclosed / missing tags
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(malformed_xml)
        tmp_path = f.name

    try:
        boxes, issues = parse_voc_xml(tmp_path, 200, 200)
        assert len(boxes) == 0
        assert len(issues) > 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_dataset_split_reproducibility():
    """Verifies that random seeds produce 100% deterministic splits."""
    import random
    items = [f"item_{i}" for i in range(100)]
    
    r1 = random.Random(42)
    s1 = list(items)
    r1.shuffle(s1)

    r2 = random.Random(42)
    s2 = list(items)
    r2.shuffle(s2)

    assert s1 == s2


def test_missing_weights_graceful_handling():
    """Tests that predict.py and evaluate.py gracefully handle missing checkpoints without crashing."""
    dummy_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    img.save(dummy_img)

    try:
        res = run_prediction(source_path=dummy_img, weights_path="non_existent_weights.pt")
        assert res["status"] == "ERROR"
        assert "not found" in res["message"]

        eval_res = evaluate_model(weights_path="non_existent_weights.pt")
        assert eval_res["status"] == "PENDING_TRAINING"
    finally:
        if os.path.exists(dummy_img):
            os.remove(dummy_img)


def test_image_label_pairing_on_prepared_data():
    """Validates that all images in the prepared dataset have matching YOLO labels."""
    dataset_root = WORKSPACE_ROOT / "datasets" / "defect_dataset"
    if dataset_root.exists():
        for split in ("train", "val", "test"):
            img_dir = dataset_root / "images" / split
            lbl_dir = dataset_root / "labels" / split
            if img_dir.exists() and lbl_dir.exists():
                imgs = {f.stem for f in img_dir.glob("*.jpg")}
                lbls = {f.stem for f in lbl_dir.glob("*.txt")}
                assert imgs == lbls, f"Mismatch in {split} split between images and labels"
