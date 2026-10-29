"""
Instant Defect Detection CLI.
Test any image directly from the terminal to detect and localize surface anomalies.
Usage:
    python detect_defect.py patches_1
    python detect_defect.py patches_1.jpg
    python detect_defect.py raw_dataset/train/images/patches/patches_1.jpg
    python detect_defect.py "C:\\Users\\arnav\\OneDrive\\Desktop\\APPLIED AI 2\\raw_dataset\\train\\images\\patches\\patches_1.jpg"
"""

import sys
import glob
from pathlib import Path
from src.predict import run_prediction

def resolve_image_path(input_arg: str) -> str:
    """Intelligently resolves image path, handling relative, absolute, stem, and space variations."""
    p = Path(input_arg)
    if p.exists() and p.is_file():
        return str(p.resolve())

    # Try searching for filename or stem across workspace
    target = p.name
    if not target.endswith((".jpg", ".jpeg", ".png")):
        target = f"{target}.jpg"

    matches = list(Path(".").glob(f"**/{target}"))
    if matches:
        return str(matches[0].resolve())

    # Return original as fallback
    return input_arg

def main():
    conf = 0.20
    image_arg = None

    args = sys.argv[1:]
    if args:
        # Check if the last argument is a float confidence
        try:
            potential_conf = float(args[-1])
            if 0.0 < potential_conf <= 1.0:
                conf = potential_conf
                args = args[:-1]
        except ValueError:
            pass

        if args:
            # Join remaining arguments in case user passed unquoted path with spaces
            raw_path = " ".join(args)
            image_arg = resolve_image_path(raw_path)

    if not image_arg:
        # Default fallback to patches_1 if available, otherwise random sample
        sample = Path("datasets/defect_dataset/images/train/patches_1.jpg")
        if not sample.exists():
            test_samples = list(Path("datasets/defect_dataset/images/test").glob("*.jpg"))
            sample = test_samples[0] if test_samples else None

        if not sample:
            print("Error: No sample images found in dataset.")
            sys.exit(1)
        image_arg = str(sample.resolve())
        print(f"[*] No image specified. Defaulting to sample: {Path(image_arg).name}")

    print(f"[*] Inspecting target: {image_arg} (confidence threshold: {conf:.2f})...")
    run_prediction(source_path=image_arg, conf_threshold=conf)

if __name__ == "__main__":
    main()
