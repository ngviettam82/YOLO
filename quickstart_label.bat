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
echo [2/2] Launching LabelImg annotation tool...
echo.
echo Instructions:
echo 1. Use arrow keys or mouse to navigate
echo 2. Click "Create RectBox" to draw bounding boxes
echo 3. Name the object with its class (e.g., 'person', 'car')
echo 4. Save after each image
echo 5. Annotations will be saved as .xml files
echo.
python scripts\label_images.py --tool labelimg
if errorlevel 1 (
    echo ERROR: Labeling tool failed!
    pause
    exit /b 1
)

echo.
echo ======================================
echo   Labeling Complete!
echo ======================================
echo.
echo Next steps:
echo 1. Update class names in dataset/data.yaml
echo.
echo 2. Start training:
echo    Double-click quickstart_train.bat
echo.
pause
