"""
Real-Time Video and Webcam Defect Detection Streamer for Industrial Inspection.
Processes video streams, live webcams, displays real-time FPS and defect tallies,
and records annotated video output when specified.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional
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


def run_video_detection(
    source: str = "0",
    weights_path: str = "outputs/baseline/best.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    output_path: Optional[str] = None,
    device: str = "auto",
    headless: bool = False,
    max_frames: Optional[int] = None,
) -> bool:
    """Streams live video or webcam inference with bounding boxes and FPS telemetry."""
    logger = setup_logger("VideoDetect")
    logger.info("Initializing Real-Time Video Defect Inspection Engine...")

    weights = Path(weights_path).resolve()
    if not weights.exists():
        alt_weights = Path("outputs/baseline/baseline_run/weights/best.pt").resolve()
        if alt_weights.exists():
            weights = alt_weights
        else:
            logger.error(
                f"Trained model checkpoint not found at '{weights_path}'. "
                "Train the custom YOLOv8 defect detection model first via 'python src/train_baseline.py'."
            )
            return False

    from ultralytics import YOLO

    model = YOLO(str(weights))
    target_device = get_best_device() if device == "auto" else device
    logger.info(f"Loaded YOLO model on device='{target_device}'")

    # Determine source (integer for webcam, string for video file)
    is_camera = source.isdigit()
    cap_source = int(source) if is_camera else source

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        if is_camera:
            logger.error(
                f"Failed to access webcam at device index {source}. "
                "Ensure a physical camera is connected and permitted by OS privacy settings, "
                "or run with a video file: python src/video_detect.py --source sample_roll.mp4"
            )
        else:
            logger.error(f"Cannot open video source file: '{source}'. Verify the file path.")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    logger.info(f"Connected video stream: {width}x{height} @ {src_fps:.1f} FPS")

    writer = None
    if output_path:
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file), fourcc, src_fps, (width, height))
        logger.info(f"Recording annotated inspection stream to: {out_file}")

    frame_count = 0
    fps_history = []
    logger.info("Starting detection loop. Press 'q' or 'ESC' in the preview window to exit.")

    try:
        while cap.isOpened():
            t_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream reached.")
                break

            frame_count += 1
            if max_frames and frame_count > max_frames:
                logger.info(f"Reached frame limit of {max_frames}.")
                break

            # YOLO inference
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                device=target_device,
                verbose=False,
            )[0]

            defect_counts = {c: 0 for c in CLASS_NAMES}
            boxes = results.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    xmin, ymin, xmax, ymax = map(int, xyxy)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = results.names.get(cls_id, f"defect_{cls_id}")
                    if cls_name in defect_counts:
                        defect_counts[cls_name] += 1

                    color = CLASS_COLORS_BGR.get(cls_name, (0, 0, 255))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                    tag = f"{cls_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (xmin, max(0, ymin - th - 6)), (xmin + tw + 4, max(0, ymin)), color, -1)
                    cv2.putText(
                        frame,
                        tag,
                        (xmin + 2, max(0, ymin - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            # Compute smoothed FPS
            loop_time = time.perf_counter() - t_start
            curr_fps = 1.0 / max(1e-4, loop_time)
            fps_history.append(curr_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Telemetry HUD Overlay
            hud_bg_w = 260
            hud_bg_h = 75
            cv2.rectangle(frame, (10, 10), (10 + hud_bg_w, 10 + hud_bg_h), (20, 24, 33), -1)
            cv2.rectangle(frame, (10, 10), (10 + hud_bg_w, 10 + hud_bg_h), (59, 130, 246), 1)

            cv2.putText(
                frame,
                f"FPS: {avg_fps:.1f} ({loop_time*1000:.0f}ms)",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 128),
                1,
                cv2.LINE_AA,
            )

            total_defects = sum(defect_counts.values())
            status_text = f"Defects: {total_defects}" if total_defects > 0 else "Surface: Normal (No Defects)"
            status_color = (0, 0, 255) if total_defects > 0 else (0, 255, 0)
            cv2.putText(
                frame,
                status_text,
                (20, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Conf: {conf_threshold:.2f} | Device: {target_device}",
                (20, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            if writer:
                writer.write(frame)

            if not headless:
                cv2.imshow("Industrial Surface Defect Inspection [Press Q to Exit]", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    logger.info("Exit command received by user.")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
            logger.info("Saved annotated video output.")
        if not headless:
            cv2.destroyAllWindows()

    logger.info(f"Inspection stream finished. Total frames processed: {frame_count}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Defect Detection on Webcam or Video")
    parser.add_argument("--source", type=str, default="0", help="Camera index ('0') or video file path")
    parser.add_argument("--weights", type=str, default="outputs/baseline/best.pt", help="Path to best.pt weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    parser.add_argument("--output", type=str, default=None, help="Optional output MP4 file path")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    parser.add_argument("--headless", action="store_true", help="Run without UI window (for batch/server)")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional maximum frames to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_video_detection(
        source=args.source,
        weights_path=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_path=args.output,
        device=args.device,
        headless=args.headless,
        max_frames=args.max_frames,
    )
