# ===================================================================
# YOLO Training Environment Setup (PowerShell)
# Optimized for RTX 5080 + CUDA 12.8
# ===================================================================

Write-Host ""
Write-Host "========================================"
Write-Host "YOLO Training Environment Setup"
Write-Host "RTX 5080 + CUDA 12.8 Optimized"
Write-Host "========================================"
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion"
    Write-Host ""
} catch {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.8-3.11 first."
    Write-Host "Download from: https://www.python.org/downloads/"
    pause
    exit 1
}

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
    Write-Host ""
    $recreate = Read-Host "Do you want to recreate it? (This will delete the existing one) [y/N]"
    if ($recreate -eq "y" -or $recreate -eq "Y") {
        Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force venv -ErrorAction SilentlyContinue
    } else {
        Write-Host "Using existing virtual environment."
        Write-Host ""
        goto skip_venv
    }
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment!" -ForegroundColor Red
    pause
    exit 1
}
Write-Host "Virtual environment created successfully." -ForegroundColor Green
Write-Host ""

:skip_venv
# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1
Write-Host ""

# Upgrade pip
Write-Host "========================================"
Write-Host "Step 1/4: Upgrading pip..." -ForegroundColor Cyan
Write-Host "========================================"
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to upgrade pip" -ForegroundColor Yellow
    Write-Host "Continuing anyway..."
}
Write-Host ""

# Install PyTorch with CUDA 12.8
Write-Host "========================================"
Write-Host "Step 2/4: Installing PyTorch with CUDA 12.8 (RTX 5080)" -ForegroundColor Cyan
Write-Host "This may take 5-10 minutes..." -ForegroundColor Yellow
Write-Host "========================================"
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install PyTorch!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try manually:"
    Write-Host "python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
    pause
    exit 1
}
Write-Host "PyTorch installed successfully." -ForegroundColor Green
Write-Host ""

# Install requirements
Write-Host "========================================"
Write-Host "Step 3/4: Installing YOLO and dependencies..." -ForegroundColor Cyan
Write-Host "This may take 3-5 minutes..." -ForegroundColor Yellow
Write-Host "========================================"
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install requirements!" -ForegroundColor Red
    pause
    exit 1
}
Write-Host "All dependencies installed successfully." -ForegroundColor Green
Write-Host ""

# Verify installation
Write-Host "========================================"
Write-Host "Step 4/4: Verifying installation..." -ForegroundColor Cyan
Write-Host "========================================"
python check_setup.py
Write-Host ""

# Final message
Write-Host "========================================"
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Prepare your dataset in the 'dataset' folder"
Write-Host "2. Create a dataset YAML file"
Write-Host "3. Run: python train_optimized.py --data dataset/your_data.yaml"
Write-Host ""
Write-Host "For detailed instructions, read:" -ForegroundColor Cyan
Write-Host "- SETUP_COMPLETE.md"
Write-Host "- RTX5080_OPTIMIZED.md"
Write-Host "- TRAINING_GUIDE.md"
Write-Host ""
Write-Host "Virtual environment is active. To deactivate, type: deactivate"
Write-Host "To reactivate later, run: .\venv\Scripts\Activate.ps1"
Write-Host ""
pause
