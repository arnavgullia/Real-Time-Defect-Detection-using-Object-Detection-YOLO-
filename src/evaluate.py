"""
Model Evaluation and Benchmarking Suite for Industrial Defect Detection.
Computes Precision, Recall, mAP50, mAP50-95, per-class metrics, confusion matrices,
FPS, and latency benchmarking on validation or held-out test splits.
"""

import os
import sys
import time
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.utils import (
    CLASS_NAMES,
    setup_logger,
    get_best_device,
    get_system_hardware_info,
)


def benchmark_latency(
    model: Any,
    img_size: int = 640,
    device: str = "cpu",
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> Dict[str, float]:
    """Measures rigorous inference latency (ms) and throughput (FPS)."""
    import torch

    dummy_input = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup_runs):
        _ = model.predict(dummy_input, device=device, verbose=False)

    latencies_ms = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy_input, device=device, verbose=False)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_latency = float(np.mean(latencies_ms))
    median_latency = float(np.median(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_latency, 2),
        "median_latency_ms": round(median_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "throughput_fps": round(fps, 1),
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
    }


def evaluate_model(
    weights_path: str = "outputs/baseline/best.pt",
    data_yaml: str = "datasets/defect_dataset/data.yaml",
    split: str = "val",
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "auto",
    output_dir: str = "outputs/reports",
) -> Dict[str, Any]:
    """Runs rigorous model validation and latency profiling."""
    logger = setup_logger("EvaluateModel", f"{output_dir}/evaluation.log")
    logger.info(f"Starting evaluation on split: '{split}'...")

    weights = Path(weights_path).resolve()
    if not weights.exists():
        # Check secondary common location
        alt_weights = Path("outputs/baseline/baseline_run/weights/best.pt").resolve()
        if alt_weights.exists():
            weights = alt_weights
        else:
            logger.warning(f"Model checkpoint not found at: {weights_path} or {alt_weights}")
            pending_report = {
                "status": "PENDING_TRAINING",
                "message": f"Checkpoint {weights_path} does not exist yet. Run src/train_baseline.py first.",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            out_root = Path(output_dir).resolve()
            out_root.mkdir(parents=True, exist_ok=True)
            with open(out_root / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(pending_report, f, indent=2)
            return pending_report

    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    model_size_mb = round(weights.stat().st_size / (1024 * 1024), 2)
    logger.info(f"Loaded YOLO weights from: {weights} ({model_size_mb} MB)")

    target_device = get_best_device() if device == "auto" else device

    # Validation
    logger.info(f"Executing model.val() on split='{split}' with imgsz={imgsz} on device='{target_device}'...")
    val_results = model.val(
        data=str(Path(data_yaml).resolve()),
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        device=target_device,
        save_json=True,
        plots=True,
        verbose=True,
    )

    # Extract metrics
    mp = float(val_results.box.mp)
    mr = float(val_results.box.mr)
    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)

    # Per-class AP50 and AP50-95
    per_class_metrics = []
    for idx, cls_name in enumerate(CLASS_NAMES):
        if idx < len(val_results.box.p):
            p_val = float(val_results.box.p[idx])
            r_val = float(val_results.box.r[idx])
            ap50 = float(val_results.box.ap50[idx]) if hasattr(val_results.box, "ap50") and idx < len(val_results.box.ap50) else 0.0
            ap = float(val_results.box.ap[idx]) if hasattr(val_results.box, "ap") and idx < len(val_results.box.ap) else 0.0
            per_class_metrics.append({
                "class_name": cls_name,
                "precision": round(p_val, 4),
                "recall": round(r_val, 4),
                "mAP50": round(ap50, 4),
                "mAP50_95": round(ap, 4),
            })

    # Benchmark Latency
    logger.info("Profiling inference latency and FPS throughput...")
    latency_stats = benchmark_latency(model, img_size=imgsz, device=target_device)

    # Check for confusion matrix plot in ultralytics run directory
    val_save_dir = Path(val_results.save_dir) if hasattr(val_results, "save_dir") else None
    confusion_matrix_path = None
    if val_save_dir and val_save_dir.exists():
        cm_src = val_save_dir / "confusion_matrix.png"
        if cm_src.exists():
            cm_dst = out_root / "confusion_matrix.png"
            shutil.copy2(cm_src, cm_dst)
            confusion_matrix_path = str(cm_dst)

    overall_metrics = {
        "status": "COMPLETED",
        "split_evaluated": split,
        "weights_path": str(weights),
        "model_file_size_mb": model_size_mb,
        "precision": round(mp, 4),
        "recall": round(mr, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "per_class": per_class_metrics,
        "latency": latency_stats,
        "confusion_matrix_plot": confusion_matrix_path,
        "hardware_info": get_system_hardware_info(),
    }

    # Save metrics.json
    with open(out_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    # Save metrics.csv
    df_metrics = pd.DataFrame([{
        "Metric": "Overall",
        "Precision": overall_metrics["precision"],
        "Recall": overall_metrics["recall"],
        "mAP50": overall_metrics["mAP50"],
        "mAP50_95": overall_metrics["mAP50_95"],
        "Mean_Latency_ms": latency_stats["mean_latency_ms"],
        "FPS": latency_stats["throughput_fps"],
        "Model_Size_MB": model_size_mb,
    }])
    if per_class_metrics:
        df_classes = pd.DataFrame(per_class_metrics).rename(columns={
            "class_name": "Metric",
            "precision": "Precision",
            "recall": "Recall",
        })
        df_combined = pd.concat([df_metrics, df_classes], ignore_index=True)
    else:
        df_combined = df_metrics
    df_combined.to_csv(out_root / "metrics.csv", index=False)

    # Save evaluation_report.md
    md_lines = [
        f"# Model Evaluation Report: YOLOv8 Baseline ({split.upper()} Split)",
        "",
        "## 1. Overall Detection Performance",
        "",
        f"- **Model Weights**: `{weights.name}` ({model_size_mb} MB)",
        f"- **Evaluation Split**: `{split}`",
        f"- **Mean Average Precision (mAP@50)**: **{overall_metrics['mAP50']:.4f}**",
        f"- **Mean Average Precision (mAP@50-95)**: **{overall_metrics['mAP50_95']:.4f}**",
        f"- **Precision (all classes)**: **{overall_metrics['precision']:.4f}**",
        f"- **Recall (all classes)**: **{overall_metrics['recall']:.4f}**",
        "",
        "## 2. Real-Time Inference Latency & Throughput",
        "",
        f"- **Target Hardware**: `{latency_stats.get('device', target_device)}`",
        f"- **Mean Latency**: **{latency_stats['mean_latency_ms']} ms** per frame",
        f"- **Median Latency**: **{latency_stats['median_latency_ms']} ms**",
        f"- **95th Percentile Latency**: **{latency_stats['p95_latency_ms']} ms**",
        f"- **Throughput**: **{latency_stats['throughput_fps']} FPS**",
        "",
        "## 3. Per-Class Detection Breakdown",
        "",
        "| Defect Category | Precision | Recall | mAP@50 | mAP@50-95 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for row in per_class_metrics:
        md_lines.append(
            f"| `{row['class_name']}` | {row['precision']:.4f} | {row['recall']:.4f} | {row['mAP50']:.4f} | {row['mAP50_95']:.4f} |"
        )

    md_lines.extend([
        "",
        "## 4. Operational Notes",
        "- The evaluation follows COCO standard object detection evaluation protocols.",
        f"- When evaluating on `{split}`, all predictions are compared against ground-truth labels from `datasets/defect_dataset/labels/{split}`.",
        "- Hyperparameters must NOT be tuned based on the test set.",
    ])

    with open(out_root / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info("Evaluation complete! Reports and metrics saved.")
    return overall_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Defect Detection Model")
    parser.add_argument("--weights", type=str, default="outputs/baseline/best.pt", help="Path to best.pt weights")
    parser.add_argument("--data", type=str, default="datasets/defect_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    parser.add_argument("--output_dir", type=str, default="outputs/reports", help="Output directory for reports")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        output_dir=args.output_dir,
    )
