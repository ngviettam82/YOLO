@echo off
REM ===================================================================
REM YOLO Training Batch Script
REM Trains YOLO model with optimized RTX 5080 settings
REM ===================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

cls
echo.
echo ======================================
echo   YOLO Training Quickstart
echo ======================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Verify dataset config exists
echo [1/3] Checking dataset configuration...
if not exist "dataset\data.yaml" (
    echo ERROR: Dataset config not found!
    echo Please run 2.dataset.bat first to prepare the dataset.
    echo.
    pause
    exit /b 1
)
echo Dataset config found: dataset/data.yaml
echo.

REM Display training options
echo [2/3] Training Configuration:
echo   Model Selection: Pretrained (yolo11m) or Load from file
echo   Training Mode: Fresh start or Resume from checkpoint
echo   Epochs: 500 (optimized configuration)
echo   Image Size: 832px (optimized configuration)
echo   Batch Size: 16 (optimized for RTX 5080)
echo   Learning Rate: 0.01 ^-^> 0.001 (optimized for stability)
echo   Patience: 50 epochs (early stopping)
echo   Optimizer: AdamW (proven stable)
echo.

REM Start training with interactive model selection
echo [3/3] Starting training...
echo This will take several hours depending on your dataset size...
echo.
echo You will be prompted to select:
echo   1. Model Source: Pretrained (yolo11m) or Load file
echo   2. Training Mode: Fresh start (epoch 1) or Resume checkpoint
echo.
echo Follow the prompts to get started...
echo.
python scripts\train_optimized.py --data dataset/data.yaml --resume
if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo ======================================
echo   Training Complete!
echo ======================================
echo.
echo Results saved to: runs/train_*/
echo.
echo Next steps:
echo 1. Validate model:
echo    python scripts\validate_model.py --model runs/train_xxx/weights/best.pt --data dataset/data.yaml
echo.
echo 2. Run inference:
echo    python scripts\inference.py --model runs/train_xxx/weights/best.pt --source test.jpg
echo.
echo 3. Export model:
echo    python scripts\export_model.py --model runs/train_xxx/weights/best.pt --formats onnx
echo.
pause
