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
echo.
echo Default Configuration:
echo   Epochs: 500
echo   Image Size: 640px
echo   Batch Size: 8
echo   Learning Rate: 0.001 ^-^> 0.0001
echo   Patience: 50 epochs (early stopping)
echo   Optimizer: SGD (numerically stable)
echo.
echo You can customize these parameters, or press ENTER to use defaults.
echo.

setlocal enabledelayedexpansion

REM Prompt for epochs
set /p EPOCHS="Enter epochs (default 500): "
if "!EPOCHS!"=="" set EPOCHS=500

REM Prompt for image size
set /p IMGSZ="Enter image size (default 640): "
if "!IMGSZ!"=="" set IMGSZ=640

REM Prompt for batch size
set /p BATCH="Enter batch size (default 8): "
if "!BATCH!"=="" set BATCH=8

REM Prompt for learning rate
set /p LR0="Enter initial learning rate (default 0.001): "
if "!LR0!"=="" set LR0=0.001

REM Prompt for patience
set /p PATIENCE="Enter patience/early stopping epochs (default 50): "
if "!PATIENCE!"=="" set PATIENCE=50

echo.
echo [3/3] Starting training with your configuration...
echo This will take several hours depending on your dataset size...
echo.
echo You will be prompted to select:
echo   1. Model Source: Pretrained (yolo11m) or Load file
echo   2. Training Mode: Fresh start (epoch 1) or Resume checkpoint
echo.
echo Configuration Summary:
echo   Epochs: !EPOCHS!
echo   Image Size: !IMGSZ!px
echo   Batch Size: !BATCH!
echo   Learning Rate: !LR0! ^-^> (final varies)
echo   Patience: !PATIENCE! epochs
echo.
python scripts\train_optimized.py --data dataset/data.yaml --epochs !EPOCHS! --imgsz !IMGSZ! --batch !BATCH! --lr0 !LR0! --patience !PATIENCE!
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
