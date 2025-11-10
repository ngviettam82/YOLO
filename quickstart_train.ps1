# Quick Training Script
# Automatically: train YOLO model with optimized RTX 5080 settings

param(
    [string]$DataConfig = "dataset/data.yaml",
    [string]$Model = "yolo11m.pt",
    [int]$Epochs = 200,
    [int]$BatchSize = 40,
    [int]$ImageSize = 832,
    [switch]$NoResume
)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  YOLO Training Quickstart" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "[1/4] Activating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "✗ Virtual environment not found. Run: python3.10 -m venv .venv" -ForegroundColor Red
    exit 1
}

# Verify dataset config exists
Write-Host ""
Write-Host "[2/4] Checking dataset configuration..." -ForegroundColor Yellow
if (-not (Test-Path $DataConfig)) {
    Write-Host "✗ Dataset config not found: $DataConfig" -ForegroundColor Red
    Write-Host "   Run: .\quickstart_dataset.ps1" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Dataset config found: $DataConfig" -ForegroundColor Green

# Display training parameters
Write-Host ""
Write-Host "[3/4] Training Configuration:" -ForegroundColor Yellow
Write-Host "  Model: $Model" -ForegroundColor White
Write-Host "  Epochs: $Epochs" -ForegroundColor White
Write-Host "  Batch Size: $BatchSize" -ForegroundColor White
Write-Host "  Image Size: $ImageSize" -ForegroundColor White
if ($NoResume) {
    Write-Host "  Resume: No (fresh start)" -ForegroundColor White
} else {
    Write-Host "  Resume: Yes (continue from last checkpoint)" -ForegroundColor White
}

# Build training command
Write-Host ""
Write-Host "[4/4] Starting training..." -ForegroundColor Yellow
Write-Host ""

$TrainArgs = @(
    "scripts/train_optimized.py",
    "--data", $DataConfig,
    "--model", $Model,
    "--epochs", $Epochs,
    "--batch", $BatchSize,
    "--imgsz", $ImageSize
)

if ($NoResume) {
    $TrainArgs += "--no-resume"
}

python @TrainArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host "  Training Complete! 🎉" -ForegroundColor Cyan
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Results saved to: runs/train_*/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Validate model:" -ForegroundColor White
    Write-Host "   python scripts/validate_model.py --model runs/train_xxx/weights/best.pt --data $DataConfig" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Run inference:" -ForegroundColor White
    Write-Host "   python scripts/inference.py --model runs/train_xxx/weights/best.pt --source test.jpg" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Export model:" -ForegroundColor White
    Write-Host "   python scripts/export_model.py --model runs/train_xxx/weights/best.pt --formats onnx" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "✗ Training failed" -ForegroundColor Red
    Write-Host "  Check error messages above" -ForegroundColor Yellow
    exit 1
}
