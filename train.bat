@echo off
REM Quick Training Script for Windows
REM Usage: train.bat

echo ========================================
echo YOLO11 Training - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install ultralytics opencv-python pyyaml
    echo.
) else (
    call venv\Scripts\activate.bat
)

echo Virtual environment activated
echo.

REM Check if dataset exists
if not exist "dataset\train" (
    echo ERROR: Dataset not found!
    echo Please prepare your dataset in the 'dataset' folder
    echo Run: python utils/dataset_utils.py validate --dataset dataset
    pause
    exit /b 1
)

REM Find dataset YAML file
set DATASET_YAML=
for %%f in (dataset\*.yaml) do set DATASET_YAML=%%f

if "%DATASET_YAML%"=="" (
    echo ERROR: No dataset YAML file found in 'dataset' folder
    echo Please create a dataset YAML file
    pause
    exit /b 1
)

echo Using dataset: %DATASET_YAML%
echo.

REM Start training
echo Starting training...
python train_optimized.py --data "%DATASET_YAML%"

echo.
echo ========================================
echo Training complete!
echo Check 'runs' folder for results
echo ========================================
pause
