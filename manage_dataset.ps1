# ===================================================================
# Dataset Preparation and Labeling Helper (PowerShell)
# Easy management for YOLO dataset preparation
# ===================================================================

param(
    [string]$Operation = "menu"
)

function Show-Menu {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "YOLO Dataset Management Tool" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Split raw dataset into train/val/test (70/20/10)" -ForegroundColor Yellow
    Write-Host "2. Split with custom ratios" -ForegroundColor Yellow
    Write-Host "3. Open annotation tool (LabelImg)" -ForegroundColor Yellow
    Write-Host "4. Create dataset configuration" -ForegroundColor Yellow
    Write-Host "5. View dataset information" -ForegroundColor Yellow
    Write-Host "6. Install annotation tools" -ForegroundColor Yellow
    Write-Host "7. Exit" -ForegroundColor Yellow
    Write-Host ""
}

function Split-DefaultDataset {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Splitting dataset (default: 70/20/10)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    & python scripts/split_dataset.py
    
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Split-CustomDataset {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Custom Dataset Split" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    $train = Read-Host "Enter training ratio (0-1, default 0.7)"
    if ([string]::IsNullOrEmpty($train)) { $train = "0.7" }
    
    $val = Read-Host "Enter validation ratio (0-1, default 0.2)"
    if ([string]::IsNullOrEmpty($val)) { $val = "0.2" }
    
    $test = Read-Host "Enter test ratio (0-1, default 0.1)"
    if ([string]::IsNullOrEmpty($test)) { $test = "0.1" }
    
    $move = Read-Host "Move files instead of copy? (y/n, default n)"
    
    if ($move -eq "y" -or $move -eq "Y") {
        & python scripts/split_dataset.py --train $train --val $val --test $test --move
    } else {
        & python scripts/split_dataset.py --train $train --val $val --test $test
    }
    
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Open-LabelingTool {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Image Labeling Tool" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    & python scripts/label_images.py
    
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function New-DatasetConfiguration {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Create Dataset Configuration" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    $classes = Read-Host "Enter number of classes (default 1)"
    if ([string]::IsNullOrEmpty($classes)) { $classes = "1" }
    
    & python scripts/label_images.py --config --num-classes $classes
    
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Show-DatasetInfo {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Dataset Information" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    $trainImages = @(Get-ChildItem "dataset\images\train\" -ErrorAction SilentlyContinue).Count
    $valImages = @(Get-ChildItem "dataset\images\val\" -ErrorAction SilentlyContinue).Count
    $testImages = @(Get-ChildItem "dataset\images\test\" -ErrorAction SilentlyContinue).Count
    
    Write-Host "Train images: $trainImages" -ForegroundColor Green
    Write-Host "Val images:   $valImages" -ForegroundColor Green
    Write-Host "Test images:  $testImages" -ForegroundColor Green
    Write-Host ""
    
    $trainLabels = @(Get-ChildItem "dataset\labels\train\*.txt" -ErrorAction SilentlyContinue).Count
    $valLabels = @(Get-ChildItem "dataset\labels\val\*.txt" -ErrorAction SilentlyContinue).Count
    $testLabels = @(Get-ChildItem "dataset\labels\test\*.txt" -ErrorAction SilentlyContinue).Count
    
    Write-Host "Train labels: $trainLabels" -ForegroundColor Green
    Write-Host "Val labels:   $valLabels" -ForegroundColor Green
    Write-Host "Test labels:  $testLabels" -ForegroundColor Green
    Write-Host ""
    
    if (Test-Path "dataset\data.yaml") {
        Write-Host "Dataset configuration file found: dataset\data.yaml" -ForegroundColor Green
    } else {
        Write-Host "WARNING: No dataset configuration file found" -ForegroundColor Yellow
        Write-Host "Run 'Create Dataset Configuration' to generate one" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Install-Tools {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Install Annotation Tools" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "1. LabelImg (Recommended - Desktop)" -ForegroundColor Yellow
    Write-Host "2. Label Studio (Web-based)" -ForegroundColor Yellow
    Write-Host "3. OpenLabeling (Fast Desktop)" -ForegroundColor Yellow
    Write-Host "4. Install All" -ForegroundColor Yellow
    Write-Host ""
    
    $choice = Read-Host "Select tool to install (1-4)"
    
    switch($choice) {
        "1" {
            Write-Host "Installing LabelImg..." -ForegroundColor Cyan
            & pip install labelimg
        }
        "2" {
            Write-Host "Installing Label Studio..." -ForegroundColor Cyan
            & pip install label-studio
        }
        "3" {
            Write-Host "Cloning OpenLabeling..." -ForegroundColor Cyan
            & git clone https://github.com/Cartucho/OpenLabeling.git
            Set-Location OpenLabeling
            & pip install -r requirements.txt
            Set-Location ..
        }
        "4" {
            Write-Host "Installing all tools..." -ForegroundColor Cyan
            & pip install labelimg label-studio
            & git clone https://github.com/Cartucho/OpenLabeling.git
            Set-Location OpenLabeling
            & pip install -r requirements.txt
            Set-Location ..
        }
        default {
            Write-Host "Invalid choice" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Read-Host "Press Enter to continue"
}

# Main loop
if ($Operation -eq "menu") {
    # Check if virtual environment exists
    if (-not (Test-Path ".venv\Scripts\activate.bat")) {
        Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
        Write-Host "Please run install.bat first" -ForegroundColor Red
        exit 1
    }
    
    # Activate virtual environment
    & .\.venv\Scripts\Activate.ps1
    
    do {
        Show-Menu
        $choice = Read-Host "Select operation (1-7)"
        
        switch($choice) {
            "1" { Split-DefaultDataset }
            "2" { Split-CustomDataset }
            "3" { Open-LabelingTool }
            "4" { New-DatasetConfiguration }
            "5" { Show-DatasetInfo }
            "6" { Install-Tools }
            "7" { 
                Write-Host ""
                Write-Host "Thank you for using YOLO Dataset Manager!" -ForegroundColor Cyan
                break
            }
            default {
                Write-Host "Invalid choice! Please try again." -ForegroundColor Red
            }
        }
    } while($choice -ne "7")
} else {
    # Direct operation mode
    & .\.venv\Scripts\Activate.ps1
    
    switch($Operation) {
        "split" { Split-DefaultDataset }
        "split-custom" { Split-CustomDataset }
        "label" { Open-LabelingTool }
        "config" { New-DatasetConfiguration }
        "info" { Show-DatasetInfo }
        "install" { Install-Tools }
        default { Show-Menu }
    }
}
