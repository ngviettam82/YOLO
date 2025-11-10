@echo off
REM ===================================================================
REM YOLO Dataset Preparation Batch Script
REM Automatically splits dataset into train/val/test and creates config
REM ===================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Check if virtual environment exists and is activated
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
echo   YOLO Dataset Quickstart
echo ======================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if raw_dataset has images
echo [1/3] Checking raw_dataset folder...
if not exist "raw_dataset" (
    echo ERROR: raw_dataset folder not found!
    echo Please create raw_dataset folder and add your images there.
    echo.
    pause
    exit /b 1
)

REM Count images in raw_dataset
setlocal enabledelayedexpansion
set "count=0"
for /r "raw_dataset" %%f in (*.jpg *.jpeg *.png *.bmp *.gif *.webp) do (
    set /a count+=1
)
if !count! equ 0 (
    echo ERROR: No images found in raw_dataset folder!
    echo Please add your images to raw_dataset/ and try again.
    echo.
    pause
    exit /b 1
)
echo Found !count! images in raw_dataset/
echo.

REM Split dataset
echo [2/3] Splitting dataset into train/val/test...
python scripts\split_dataset.py --train 0.7 --val 0.2 --test 0.1
if errorlevel 1 (
    echo ERROR: Dataset splitting failed!
    pause
    exit /b 1
)
echo Dataset split complete!
echo.

REM Create dataset config
echo [3/3] Creating dataset configuration...
python scripts\label_images.py --config --num-classes 3
if errorlevel 1 (
    echo WARNING: Config creation had issues, but dataset is ready
    echo You can manually create dataset/data.yaml if needed
)

REM Summary
echo.
echo ======================================
echo   Dataset Ready!
echo ======================================
echo.
echo Next steps:
echo 1. Label your training images:
echo    Double-click 3.label.bat
echo.
echo 2. Update class names in dataset/data.yaml
echo.
echo 3. Start training:
echo    Double-click 4.train.bat
echo.
pause
