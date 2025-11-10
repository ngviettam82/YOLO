@echo off
REM ===================================================================
REM YOLO Image Labeling Tool Batch Script
REM Opens annotation tool for labeling training images
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
echo   YOLO Image Labeling Tool
echo ======================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if images exist
echo [1/2] Checking for training images...
if not exist "dataset\images\train" (
    echo ERROR: No training images folder found!
    echo Please run quickstart_dataset.bat first to prepare the dataset.
    echo.
    pause
    exit /b 1
)

setlocal enabledelayedexpansion
set "count=0"
for /r "dataset\images\train" %%f in (*.jpg *.jpeg *.png *.bmp *.gif *.webp) do (
    set /a count+=1
)
if !count! equ 0 (
    echo ERROR: No images found in dataset/images/train/
    echo Please run quickstart_dataset.bat first to prepare the dataset.
    echo.
    pause
    exit /b 1
)
echo Found !count! images ready for labeling
echo.

REM Launch annotation tool
echo [2/2] Launching annotation tool...
echo.

REM Set environment variables for Label Studio to handle large file uploads
set DJANGO_DATA_UPLOAD_MAX_NUMBER_FILES=10000
set LABEL_STUDIO_DATA_UPLOAD_MAX_NUMBER_FILES=10000

REM Default to Label Studio (web-based, reliable, no desktop GUI issues)
echo Using Label Studio as the default annotation tool
echo.
python scripts\label_images.py --tool label-studio
if errorlevel 1 (
    echo.
    echo ERROR: Label Studio failed to launch!
    echo.
    echo Troubleshooting options:
    echo 1. Install Label Studio manually:
    echo    .venv\Scripts\activate.bat
    echo    pip install label-studio
    echo.
    echo 2. Start Label Studio manually:
    echo    .venv\Scripts\activate.bat
    echo    label-studio
    echo.
    echo 3. Open browser to: http://localhost:8080
    echo.
    echo 4. Alternative: Use Roboflow (AI-assisted, cloud-based)
    echo    python scripts\label_images.py --tool roboflow
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================
echo   Labeling Complete!
echo ======================================
echo.
echo Next steps:
echo 1. Export annotations from Label Studio as YOLO format
echo.
echo 2. Place .txt files in: dataset\labels\train\
echo.
echo 3. Update class names in dataset/data.yaml
echo.
echo 4. Start training:
echo    Double-click 4.train.bat
echo.
pause
