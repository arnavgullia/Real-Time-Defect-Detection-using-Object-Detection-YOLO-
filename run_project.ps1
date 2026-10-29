# ==============================================================================
# End-to-End Execution Script: Real-Time Industrial Defect Detection System
# PowerShell Automation for Windows Environments
# ==============================================================================

param (
    [string]$Action = "all"
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   ApexDefect AI: Industrial Quality Inspection Pipeline       " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# Step 1: Environment Check
Write-Host "`n[1/7] Checking Python Environment..." -ForegroundColor Yellow
python -c "import sys; print('Python Runtime:', sys.version.split()[0])"

# Step 2: Prepare Dataset
if ($Action -eq "all" -or $Action -eq "prepare") {
    Write-Host "`n[2/7] Running XML-to-YOLO Conversion (src/prepare_dataset.py)..." -ForegroundColor Yellow
    python src/prepare_dataset.py
}

# Step 3: Validate Dataset
if ($Action -eq "all" -or $Action -eq "validate") {
    Write-Host "`n[3/7] Validating Converted Bounding Boxes (src/validate_dataset.py)..." -ForegroundColor Yellow
    python src/validate_dataset.py
}

# Step 4: Exploratory Data Analysis
if ($Action -eq "all" -or $Action -eq "analyze") {
    Write-Host "`n[4/7] Generating EDA Analytics & Size Charts (src/analyze_dataset.py)..." -ForegroundColor Yellow
    python src/analyze_dataset.py
}

# Step 5: Unit Tests
if ($Action -eq "all" -or $Action -eq "test") {
    Write-Host "`n[5/7] Executing Automated PyTest Suite..." -ForegroundColor Yellow
    python -m pytest tests/test_pipeline.py -v
}

# Step 6: Baseline Training
if ($Action -eq "train") {
    Write-Host "`n[6/7] Training YOLOv8 Baseline Model (src/train_baseline.py)..." -ForegroundColor Yellow
    python src/train_baseline.py --epochs 50 --imgsz 640 --batch 16
}

# Step 7: Model Evaluation
if ($Action -eq "evaluate") {
    Write-Host "`n[7/7] Evaluating Trained Baseline Checkpoint (src/evaluate.py)..." -ForegroundColor Yellow
    python src/evaluate.py --weights outputs/baseline/best.pt --split val
}

# Step 8: Terminal Image Defect Detection
if ($Action -eq "detect" -or $Action -eq "inspect") {
    Write-Host "`nRunning Terminal Defect Detection (detect_defect.py)..." -ForegroundColor Green
    python detect_defect.py
}

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host " Pipeline execution completed successfully. " -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
