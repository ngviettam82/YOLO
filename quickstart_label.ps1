# Quick Label Images Script
# Opens annotation tool for labeling training images

param(
    [string]$Tool = "labelimg",
    [string]$ImageFolder = "dataset/images/train"
)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  YOLO Image Labeling Tool" -ForegroundColor Cyan
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

# Check if images exist
Write-Host ""
Write-Host "[2/3] Checking for images in $ImageFolder..." -ForegroundColor Yellow
$ImageCount = @(Get-ChildItem $ImageFolder -Include *.jpg, *.jpeg, *.png, *.bmp, *.gif, *.webp -ErrorAction SilentlyContinue).Count
if ($ImageCount -eq 0) {
    Write-Host "✗ No images found in $ImageFolder" -ForegroundColor Red
    Write-Host "   Run: .\quickstart_dataset.ps1 (to split dataset first)" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "✓ Found $ImageCount images in $ImageFolder" -ForegroundColor Green
}

# Launch annotation tool
Write-Host ""
Write-Host "[3/3] Launching $Tool..." -ForegroundColor Yellow
Write-Host ""

switch ($Tool.ToLower()) {
    "labelimg" {
        Write-Host "Opening LabelImg for annotation..." -ForegroundColor Cyan
        Write-Host "Instructions:" -ForegroundColor Yellow
        Write-Host "1. Use WASD keys or arrow keys to navigate" -ForegroundColor Gray
        Write-Host "2. Click 'Create RectBox' to draw bounding boxes" -ForegroundColor Gray
        Write-Host "3. Name the object with its class (e.g., 'person', 'car')" -ForegroundColor Gray
        Write-Host "4. Save after each image" -ForegroundColor Gray
        Write-Host "5. Annotations will be saved as .xml files" -ForegroundColor Gray
        Write-Host ""
        python scripts/label_images.py --tool labelimg
    }
    "cvat" {
        Write-Host "Opening CVAT (web-based)..." -ForegroundColor Cyan
        Write-Host "CVAT will open in your default browser" -ForegroundColor Yellow
        python scripts/label_images.py --tool cvat
    }
    "label-studio" {
        Write-Host "Opening Label Studio (web-based)..." -ForegroundColor Cyan
        Write-Host "Label Studio will open in your default browser" -ForegroundColor Yellow
        python scripts/label_images.py --tool label-studio
    }
    "openlabeling" {
        Write-Host "Opening OpenLabeling..." -ForegroundColor Cyan
        Write-Host "Instructions:" -ForegroundColor Yellow
        Write-Host "1. Optimized for fast annotation" -ForegroundColor Gray
        Write-Host "2. Use click and drag to create boxes" -ForegroundColor Gray
        Write-Host "3. Type class name and press Enter" -ForegroundColor Gray
        python scripts/label_images.py --tool openlabeling
    }
    "roboflow" {
        Write-Host "Opening Roboflow (cloud AI-assisted)..." -ForegroundColor Cyan
        Write-Host "Roboflow will open in your default browser" -ForegroundColor Yellow
        python scripts/label_images.py --tool roboflow
    }
    default {
        Write-Host "Unknown tool: $Tool" -ForegroundColor Red
        Write-Host ""
        Write-Host "Available tools:" -ForegroundColor Cyan
        Write-Host "1. labelimg      - Fast desktop tool (Recommended)" -ForegroundColor Gray
        Write-Host "2. cvat          - Professional web-based" -ForegroundColor Gray
        Write-Host "3. label-studio  - Easy web-based" -ForegroundColor Gray
        Write-Host "4. openlabeling  - Speed-optimized desktop" -ForegroundColor Gray
        Write-Host "5. roboflow      - Cloud AI-assisted" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Usage: .\quickstart_label.ps1 -Tool labelimg" -ForegroundColor Yellow
        exit 1
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host "  Labeling Complete! 🎉" -ForegroundColor Cyan
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Create dataset config:" -ForegroundColor White
    Write-Host "   python scripts/label_images.py --config --num-classes 3" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Update class names in dataset/data.yaml" -ForegroundColor White
    Write-Host ""
    Write-Host "3. Start training:" -ForegroundColor White
    Write-Host "   .\quickstart_train.ps1" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "✗ Labeling tool failed" -ForegroundColor Red
    Write-Host "  Check error messages above" -ForegroundColor Yellow
    exit 1
}
