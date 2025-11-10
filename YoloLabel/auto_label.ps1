#!/usr/bin/env pwsh
# ===================================================================
# Auto-Label Images with Pre-trained YOLO Model
# ===================================================================

Set-Location $PSScriptRoot
Set-Location ".."

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run 1.install.bat first to set up the environment."
    Read-Host "Press Enter to exit"
    exit 1
}

Clear-Host
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "   Auto-Label with Pre-trained YOLO" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
& ".venv\Scripts\Activate.ps1"

# Check if images exist
Write-Host "[1/3] Checking for training images..." -ForegroundColor Yellow

if (-not (Test-Path "dataset\images\train")) {
    Write-Host "ERROR: No training images folder found!" -ForegroundColor Red
    Write-Host "Please run 2.dataset.bat first to prepare the dataset."
    Read-Host "Press Enter to exit"
    exit 1
}

$imageCount = @(Get-ChildItem "dataset\images\train" -Include *.jpg, *.jpeg, *.png, *.bmp, *.webp).Count
if ($imageCount -eq 0) {
    Write-Host "ERROR: No images found in dataset/images/train/" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Found $imageCount images ready for auto-labeling" -ForegroundColor Green
Write-Host ""

# Check if required packages are installed
Write-Host "[2/3] Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import cv2" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "cv2 not found"
    }
} catch {
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    pip install -q opencv-python ultralytics tqdm
}

# Run auto-labeling
Write-Host "[3/3] Auto-labeling images with pre-trained YOLO..." -ForegroundColor Yellow
Write-Host ""
Write-Host "This will use yolo11m.pt to detect objects" -ForegroundColor Cyan
Write-Host "Default confidence threshold: 0.5" -ForegroundColor Cyan
Write-Host ""

python YoloLabel\auto_label.py `
    --images dataset\images\train `
    --output dataset\labels\train `
    --model yolo11m.pt `
    --conf 0.5 `
    --visualize `
    --viz-limit 50

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Auto-labeling failed!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "   Auto-Labeling Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "✅ Results:" -ForegroundColor Green
Write-Host "   - Labels saved to: dataset\labels\train\" -ForegroundColor Green
Write-Host "   - Visualizations: dataset\visualizations\" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Review auto-generated labels:" -ForegroundColor Yellow
Write-Host "   - Check visualizations folder for sample images with bboxes" -ForegroundColor Yellow
Write-Host "   - Open dataset\labels\train\ to review .txt files" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Use Label Studio to verify and correct:" -ForegroundColor Yellow
Write-Host "   - Run: 3.label.bat" -ForegroundColor Yellow
Write-Host "   - Import generated labels" -ForegroundColor Yellow
Write-Host "   - Fix any false positives/negatives" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. After correction, start training:" -ForegroundColor Yellow
Write-Host "   - Run: 4.train.bat" -ForegroundColor Yellow
Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to exit"
