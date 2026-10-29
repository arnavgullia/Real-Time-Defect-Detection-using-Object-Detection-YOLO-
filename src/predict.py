"""
Inference Script for Industrial Defect Detection on Single Images or Batch Directories.
Supports confidence thresholds, IoU filtering, visualization export, and structured JSON results.
"""

import os
import sys
import glob
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.utils import (
    CLASS_NAMES,
    CLASS_COLORS_BGR,
    setup_logger,
    get_best_device,
)


def run_prediction(
    source_path: str,
    weights_path: str = "outputs/baseline/best.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    output_dir: str = "outputs/predictions",
    device: str = "auto",
    save_txt: bool = False,
) -> Dict[str, Any]:
    """Runs YOLOv8 defect detection on images."""
    logger = setup_logger("Predict", f"{output_dir}/predict.log")
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    weights = Path(weights_path).resolve()
    if not weights.exists():
        alt_weights = Path("outputs/baseline/baseline_run/weights/best.pt").resolve()
        if alt_weights.exists():
            weights = alt_weights
        else:
            msg = (
                f"Trained model checkpoint not found at: {weights_path}. "
                "The custom industrial defect detection model must be trained first using 'python src/train_baseline.py'."
            )
            logger.error(msg)
            return {"status": "ERROR", "message": msg, "predictions": []}

    from ultralytics import YOLO

    model = YOLO(str(weights))
    logger.info(f"Loaded weights from {weights}")

    target_device = get_best_device() if device == "auto" else device

    src = Path(source_path).resolve()
    if src.is_file():
        image_files = [src]
    elif src.is_dir():
        image_files = sorted([
            f for f in src.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ])
    else:
        # Try glob pattern
        image_files = sorted([Path(f) for f in glob.glob(source_path)])

    if not image_files:
        logger.warning(f"No valid images found at source: {source_path}")
        return {"status": "NO_IMAGES", "predictions": []}

    logger.info(f"Running inference on {len(image_files)} image(s) with conf={conf_threshold}, iou={iou_threshold}...")

    all_predictions = []
    total_time = 0.0

    for img_file in image_files:
        t0 = time.perf_counter()
        results = model.predict(
            source=str(img_file),
            conf=conf_threshold,
            iou=iou_threshold,
            device=target_device,
            verbose=False,
        )[0]
        inference_time_ms = (time.perf_counter() - t0) * 1000.0
        total_time += inference_time_ms

        orig_img = cv2.imread(str(img_file))
        if orig_img is None:
            continue

        detections = []
        boxes = results.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = map(int, xyxy)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = results.names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "box": [xmin, ymin, xmax, ymax],
                })

                # Visual annotation with high contrast
                bgr_color = CLASS_COLORS_BGR.get(cls_name, (0, 0, 255))
                cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), bgr_color, 2)

                label_str = f"{cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(orig_img, (xmin, max(0, ymin - th - 6)), (xmin + tw + 4, max(0, ymin)), bgr_color, -1)
                cv2.putText(
                    orig_img,
                    label_str,
                    (xmin + 2, max(0, ymin - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Save annotated image
        dest_img_path = out_path / f"pred_{img_file.name}"
        cv2.imwrite(str(dest_img_path), orig_img)

        # Save YOLO txt if requested
        if save_txt:
            dest_txt = out_path / f"pred_{img_file.stem}.txt"
            h, w = orig_img.shape[:2]
            with open(dest_txt, "w", encoding="utf-8") as tf:
                for d in detections:
                    bx = d["box"]
                    bw = bx[2] - bx[0]
                    bh = bx[3] - bx[1]
                    bxc = (bx[0] + bw / 2.0) / w
                    byc = (bx[1] + bh / 2.0) / h
                    tf.write(f"{d['class_id']} {bxc:.6f} {byc:.6f} {bw/w:.6f} {bh/h:.6f} {d['confidence']:.4f}\n")

        # Terminal formatted inspection report
        print(f"\n=======================================================")
        print(f"  SURFACE INSPECTION: {img_file.name}")
        print(f"=======================================================")
        if len(detections) > 0:
            print(f"  DECISION : [REJECT] - {len(detections)} Surface Defect(s) Detected!")
            print(f"  Latency  : {inference_time_ms:.1f} ms")
            print(f"  -----------------------------------------------------")
            print(f"  Detected Anomalies:")
            for idx, d in enumerate(detections, 1):
                conf_pct = d['confidence'] * 100.0
                print(f"   {idx}. Class: {d['class_name']:<16} | Confidence: {conf_pct:5.1f}% | BBox: {d['box']}")
        else:
            print(f"  DECISION : [PASS] - Normal Surface (No Defects Found)")
            print(f"  Latency  : {inference_time_ms:.1f} ms")
        print(f"  Annotated: {dest_img_path}")
        print(f"=======================================================")

        all_predictions.append({
            "image": img_file.name,
            "detections_count": len(detections),
            "inference_time_ms": round(inference_time_ms, 2),
            "detections": detections,
            "annotated_path": str(dest_img_path),
        })

    summary = {
        "status": "SUCCESS",
        "total_images": len(image_files),
        "mean_latency_ms": round(total_time / max(1, len(image_files)), 2),
        "predictions": all_predictions,
    }

    with open(out_path / "prediction_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Predictions completed: {len(image_files)} images processed. "
        f"Mean latency: {summary['mean_latency_ms']} ms. Outputs in: {out_path}"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 Defect Detection on Images")
    parser.add_argument("--source", type=str, default=None, help="Path to image file or directory (defaults to test set sample)")
    parser.add_argument("--weights", type=str, default="outputs/baseline/best.pt", help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions", help="Directory to save predictions")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    parser.add_argument("--save_txt", action="store_true", help="Save YOLO formatted detection labels")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # If no source provided, default to an authentic sample from the test set
    source_to_use = args.source
    if not source_to_use:
        default_candidates = list(Path("datasets/defect_dataset/images/test").glob("*.jpg"))
        if default_candidates:
            source_to_use = str(default_candidates[0])
            print(f"[*] No --source specified. Testing default sample: {source_to_use}")
        else:
            raise FileNotFoundError("No image source provided and test directory is empty.")

    run_prediction(
        source_path=source_to_use,
        weights_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_dir=args.output_dir,
        device=args.device,
        save_txt=args.save_txt,
    )

