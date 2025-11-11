@echo off
REM ===================================================================
REM STEP 1/4: YOLO Training Environment Setup Batch Installer
REM ===================================================================
REM This is the FIRST file to run!
REM It sets up Python 3.10, virtual environment, and all dependencies
REM Optimized for RTX 5080 + CUDA 12.8
REM ===================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

cls
echo.
echo ========================================
echo STEP 1 of 4: YOLO Environment Setup
echo ========================================
echo RTX 5080 + CUDA 12.8 Optimized
echo.
echo This script will:
echo  1. Check for Python 3.10
echo  2. Create virtual environment
echo  3. Install PyTorch with CUDA support
echo  4. Install all dependencies
echo.
echo ========================================
echo.

REM Check if Python 3.10 is available
echo Checking for Python 3.10...
python3.10 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.10 not found!
    echo Please install Python 3.10 first.
    echo Download from: https://www.python.org/downloads/release/python-3100/
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python3.10 --version') do set PYVER=%%i
echo %PYVER% found
echo.

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo Virtual environment already exists.
    echo.
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo Removing old virtual environment...
        rmdir /s /q .venv
    ) else (
        echo Using existing virtual environment.
        echo.
        goto skip_venv_creation
    )
)

REM Create virtual environment with Python 3.10
echo Creating virtual environment with Python 3.10...
python3.10 -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)
echo Virtual environment created successfully.
echo.

:skip_venv_creation
REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo ========================================
echo Step 1/4: Upgrading pip...
echo ========================================
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip
    echo Continuing anyway...
)
echo.

REM Install PyTorch with CUDA 12.8
echo ========================================
echo Step 2/4: Installing PyTorch with CUDA 12.8 (RTX 5080)
echo This may take 5-10 minutes...
echo ========================================
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch!
    echo.
    echo Try manually:
    echo python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pause
    exit /b 1
)
echo PyTorch installed successfully.
echo.

REM Create dataset folder structure
echo ========================================
echo Step 3/4: Creating dataset folder structure...
echo ========================================
if not exist "dataset" mkdir dataset
if not exist "dataset\images" mkdir dataset\images
if not exist "dataset\images\train" mkdir dataset\images\train
if not exist "dataset\images\val" mkdir dataset\images\val
if not exist "dataset\images\test" mkdir dataset\images\test
if not exist "dataset\labels" mkdir dataset\labels
if not exist "dataset\labels\train" mkdir dataset\labels\train
if not exist "dataset\labels\val" mkdir dataset\labels\val
if not exist "dataset\labels\test" mkdir dataset\labels\test
if not exist "raw_dataset" mkdir raw_dataset
echo Folder structure created.
echo.

REM Install requirements
echo ========================================
echo Step 4/4: Installing YOLO and dependencies...
echo This may take 3-5 minutes...
echo ========================================
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements!
    pause
    exit /b 1
)
echo All dependencies installed successfully.
echo.

REM Verify installation
echo ========================================
echo Step 5/5: Verifying installation...
echo ========================================
python scripts\check_setup.py
echo.

REM Final message
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Place your dataset images in: raw_dataset/
echo 2. Prepare your dataset: Double-click 2.dataset.bat
echo 3. Label images: Double-click 3.label.bat
echo 4. Train model: Double-click 4.train.bat
echo.
echo For detailed instructions, read:
echo - docs/INSTALLATION.md
echo - docs/DATASET_GUIDE.md
echo - docs/TRAINING_GUIDE.md
echo.
echo Virtual environment is active. To deactivate, type: deactivate
echo To reactivate later, run: .venv\Scripts\activate.bat
echo.
pause
