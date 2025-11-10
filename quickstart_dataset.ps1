# Quick Dataset Preparation Script
# Automatically: split dataset, create config, and prepare for training

param(
    [int]$NumClasses = 3,
    [double]$TrainRatio = 0.7,
    [double]$ValRatio = 0.2,
    [double]$TestRatio = 0.1
)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  YOLO Dataset Quickstart" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "[1/3] Activating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "✗ Virtual environment not found. Run: python3.10 -m venv .venv" -ForegroundColor Red
    exit 1
}

# Check if raw_dataset has images
Write-Host ""
Write-Host "[2/3] Checking raw_dataset folder..." -ForegroundColor Yellow
$ImageCount = @(Get-ChildItem "raw_dataset" -Include *.jpg, *.jpeg, *.png, *.bmp, *.gif, *.webp -ErrorAction SilentlyContinue).Count
if ($ImageCount -eq 0) {
    Write-Host "✗ No images found in raw_dataset/" -ForegroundColor Red
    Write-Host "   Add your images to raw_dataset/ folder and try again" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "✓ Found $ImageCount images in raw_dataset/" -ForegroundColor Green
}

# Split dataset
Write-Host ""
Write-Host "[3/3] Splitting dataset into train/val/test..." -ForegroundColor Yellow
python scripts/split_dataset.py --train $TrainRatio --val $ValRatio --test $TestRatio

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Dataset splitting failed" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Dataset split complete" -ForegroundColor Green

# Create dataset config
Write-Host ""
Write-Host "[Bonus] Creating dataset configuration..." -ForegroundColor Yellow
python scripts/label_images.py --config --num-classes $NumClasses

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dataset config created at dataset/data.yaml" -ForegroundColor Green
} else {
    Write-Host "⚠ Config creation had issues, but dataset is ready" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Dataset Ready! 🎉" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Label your training images:" -ForegroundColor White
Write-Host "   .\quickstart_label.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Update class names in dataset/data.yaml" -ForegroundColor White
Write-Host ""
Write-Host "3. Start training:" -ForegroundColor White
Write-Host "   .\quickstart_train.ps1" -ForegroundColor Gray
Write-Host ""
