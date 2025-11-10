@echo off
REM ===================================================================
REM YOLO Training Environment Setup
REM Optimized for RTX 5080 + CUDA 12.8
REM ===================================================================

echo.
echo ========================================
echo YOLO Training Environment Setup
echo RTX 5080 + CUDA 12.8 Optimized
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8-3.11 first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Virtual environment already exists.
    echo.
    choice /C YN /M "Do you want to recreate it? (This will delete the existing one)"
    if errorlevel 2 goto skip_venv
    if errorlevel 1 (
        echo Removing old virtual environment...
        rmdir /s /q venv
    )
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)
echo Virtual environment created successfully.
echo.

:skip_venv
REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
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

REM Install requirements
echo ========================================
echo Step 3/4: Installing YOLO and dependencies...
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
echo Step 4/4: Verifying installation...
echo ========================================
python check_setup.py
echo.

REM Final message
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Prepare your dataset in the 'dataset' folder
echo 2. Create a dataset YAML file
echo 3. Run: python train_optimized.py --data dataset/your_data.yaml
echo.
echo For detailed instructions, read:
echo - SETUP_COMPLETE.md
echo - RTX5080_OPTIMIZED.md
echo - TRAINING_GUIDE.md
echo.
echo Virtual environment is active. To deactivate, type: deactivate
echo To reactivate later, run: venv\Scripts\activate.bat
echo.
pause
