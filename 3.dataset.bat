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
echo   YOLO Dataset Preparation
echo ======================================
echo.
echo This script will:
echo   1. Check raw_dataset for labeled images
echo   2. Split images into train/val/test
echo   3. Copy labels for train and val only
echo   4. Create dataset configuration
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if raw_dataset has images
echo [1/4] Checking raw_dataset folder...
if not exist "raw_dataset" (
    echo ERROR: raw_dataset folder not found!
    echo Please create raw_dataset folder and add your labeled images there.
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
    echo Please add your labeled images to raw_dataset/ and try again.
    echo.
    pause
    exit /b 1
)
echo Found !count! images in raw_dataset/

REM Count label files
set "label_count=0"
for /r "raw_dataset" %%f in (*.txt) do (
    set /a label_count+=1
)
echo Found !label_count! label files in raw_dataset/
echo.

REM Split dataset
echo [2/5] Splitting dataset into train/val/test...
echo   - Train set: 70%% (with labels)
echo   - Val set:   20%% (with labels)
echo   - Test set:  10%% (no labels)
echo.
python scripts\split_dataset.py --train 0.7 --val 0.2 --test 0.1
if errorlevel 1 (
    echo ERROR: Dataset splitting failed!
    pause
    exit /b 1
)
echo Dataset split complete!
echo.

REM Convert Label Studio notes.json to data.yaml
echo [3/5] Converting Label Studio notes.json to data.yaml...
python scripts\convert_labels.py --raw-dataset raw_dataset --output dataset
if errorlevel 1 (
    echo WARNING: Could not convert notes.json, creating default config
    python scripts\label_images.py --config --num-classes 2
)
echo.

REM Summary
echo.
echo [4/5] Summary
echo ======================================
echo   Dataset Prepared Successfully!
echo ======================================
echo.
echo Structure:
echo   dataset/images/train/  - Training images with labels
echo   dataset/images/val/    - Validation images with labels
echo   dataset/images/test/   - Test images (no labels)
echo.
echo   dataset/labels/train/  - Training labels (copied from raw_dataset)
echo   dataset/labels/val/    - Validation labels (copied from raw_dataset)
echo   dataset/labels/test/   - Empty (no labels for test set)
echo.
echo   dataset/data.yaml      - Dataset configuration (from notes.json)
echo.
echo Next steps:
echo 1. Review dataset/data.yaml if needed
echo.
echo 2. Start training:
echo    Double-click 4.train.bat
echo.
echo [5/5] Complete!
pause
