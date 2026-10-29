"""
Utility functions and constants for the Real-Time Industrial Defect Detection System.
Supports NEU-DET steel surface defect detection.
"""

import os
import sys
import json
import logging
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml
import torch

# Standard Class Names and ID mapping for NEU-DET
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Harmonious visual palette for annotations (RGB for matplotlib/PIL, BGR for cv2)
CLASS_COLORS_RGB = {
    "crazing": (239, 68, 68),          # Vibrant Coral Red
    "inclusion": (249, 115, 22),       # Amber Orange
    "patches": (234, 179, 8),          # Golden Yellow
    "pitted_surface": (16, 185, 129),   # Emerald Green
    "rolled-in_scale": (59, 130, 246),  # Royal Blue
    "scratches": (168, 85, 247),       # Purple Violet
}

CLASS_COLORS_BGR = {
    name: (rgb[2], rgb[1], rgb[0]) for name, rgb in CLASS_COLORS_RGB.items()
}


def setup_logger(name: str = "DefectDetection", log_file: Optional[str] = None) -> logging.Logger:
    """Configures a standardized console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers on re-entry
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file safely."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """Saves a dictionary to a YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def voc_to_yolo_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float, img_w: float, img_h: float
) -> Tuple[float, float, float, float]:
    """
    Converts Pascal VOC bounding box [xmin, ymin, xmax, ymax]
    into normalized YOLO format [x_center, y_center, width, height].
    Clamps bounds safely within [0, 1].
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image dimensions: {img_w}x{img_h}")

    # Clamp coordinates to image boundaries
    xmin_clamped = max(0.0, min(float(xmin), float(img_w)))
    ymin_clamped = max(0.0, min(float(ymin), float(img_h)))
    xmax_clamped = max(0.0, min(float(xmax), float(img_w)))
    ymax_clamped = max(0.0, min(float(ymax), float(img_h)))

    box_w = max(0.0, xmax_clamped - xmin_clamped)
    box_h = max(0.0, ymax_clamped - ymin_clamped)

    if box_w <= 0 or box_h <= 0:
        raise ValueError(f"Zero or negative bounding box area: [{xmin}, {ymin}, {xmax}, {ymax}]")

    x_center = (xmin_clamped + box_w / 2.0) / float(img_w)
    y_center = (ymin_clamped + box_h / 2.0) / float(img_h)
    norm_w = box_w / float(img_w)
    norm_h = box_h / float(img_h)

    # Final normalization clamping to [0.0, 1.0]
    return (
        round(max(0.0, min(1.0, x_center)), 6),
        round(max(0.0, min(1.0, y_center)), 6),
        round(max(0.0, min(1.0, norm_w)), 6),
        round(max(0.0, min(1.0, norm_h)), 6),
    )


def yolo_to_voc_bbox(
    x_center: float, y_center: float, norm_w: float, norm_h: float, img_w: float, img_h: float
) -> Tuple[int, int, int, int]:
    """Converts normalized YOLO format back to pixel coordinates [xmin, ymin, xmax, ymax]."""
    xc = float(x_center) * float(img_w)
    yc = float(y_center) * float(img_h)
    w = float(norm_w) * float(img_w)
    h = float(norm_h) * float(img_h)

    xmin = int(round(max(0.0, xc - w / 2.0)))
    ymin = int(round(max(0.0, yc - h / 2.0)))
    xmax = int(round(min(float(img_w), xc + w / 2.0)))
    ymax = int(round(min(float(img_h), yc + h / 2.0)))

    return xmin, ymin, xmax, ymax


def get_system_hardware_info() -> Dict[str, Any]:
    """Gathers detailed hardware and environment configuration."""
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    device_count = torch.cuda.device_count() if cuda_avail else 0

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "device_count": device_count,
    }


def get_best_device() -> str:
    """Returns 'cuda:0' if CUDA is accessible, otherwise 'cpu'."""
    return "0" if torch.cuda.is_available() else "cpu"
