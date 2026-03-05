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
echo This script helps you label images in raw_dataset/
echo Labels will be saved alongside images as .txt files
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if raw_dataset exists
echo [1/2] Checking for raw images...
if not exist "raw_dataset" (
    echo ERROR: raw_dataset folder not found!
    echo Please create raw_dataset folder and add your images there.
    echo.
    pause
    exit /b 1
)

setlocal enabledelayedexpansion
set "count=0"
for /r "raw_dataset" %%f in (*.jpg *.jpeg *.png *.bmp *.gif *.webp) do (
    set /a count+=1
)
if !count! equ 0 (
    echo ERROR: No images found in raw_dataset/
    echo Please add your images to raw_dataset/ first.
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
REM Label Studio reads DATA_UPLOAD_MAX_NUMBER_FILES from Django settings
set DATA_UPLOAD_MAX_NUMBER_FILES=10000
set DJANGO_DATA_UPLOAD_MAX_NUMBER_FILES=10000
set LABEL_STUDIO_DATA_UPLOAD_MAX_NUMBER_FILES=10000
set DJANGO_FILE_UPLOAD_MAX_MEMORY_SIZE=52428800
set DATA_UPLOAD_MAX_MEMORY_SIZE=52428800
set LABEL_STUDIO_DATA_UPLOAD_MAX_MEMORY_SIZE=52428800
set CLIENT_MAX_BODY_SIZE=200m
set NGINX_CLIENT_MAX_BODY_SIZE=200m
set UWSGI_HTTP_TIMEOUT=600

REM For very large datasets (1000+ images), add extra memory settings
if !count! geq 1000 (
    set LABEL_STUDIO_WEB_LOCKED_UI=false
    echo Note: Using extended settings for large dataset
)

REM Warn about large datasets and suggest server-based import
if !count! geq 200 (
    echo.
    echo =====================================================================
    echo   LARGE DATASET DETECTED: !count! images
    echo =====================================================================
    echo.
    echo   Direct file upload may fail with "DATA_UPLOAD_MAX_NUMBER_FILES"
    echo   error for large datasets.
    echo.
    echo   RECOMMENDED: Use the server-based import instead:
    echo     1. First auto-label:  AutoLabel\run_auto_label.bat
    echo     2. Then import:       AutoLabel\import_to_label_studio.bat
    echo.
    echo   This serves images via HTTP ^(no upload limit^) and is more reliable.
    echo.
    echo   Press any key to continue with Label Studio anyway, or Ctrl+C to cancel.
    echo =====================================================================
    pause >nul
)

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
echo Your labels are saved in raw_dataset/ as .txt files
echo.
echo Next steps:
echo 1. Run 2.dataset.bat to split labeled data into train/val/test
echo    This will copy images and labels to dataset/ folder
echo.
echo 2. Start training:
echo    Double-click 4.train.bat
echo.
pause
