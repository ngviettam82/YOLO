@echo off
REM ===================================================================
REM Auto-Label Images with Pre-trained YOLO Model
REM ===================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0\.."

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run 1.install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

cls
echo.
echo ======================================
echo   Auto-Label with Pre-trained YOLO
echo ======================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if images exist
echo [1/3] Checking for training images...
if not exist "dataset\images\train" (
    echo ERROR: No training images folder found!
    echo Please run 2.dataset.bat first to prepare the dataset.
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
    echo.
    pause
    exit /b 1
)
echo Found !count! images ready for auto-labeling
echo.

REM Check if required packages are installed
echo [2/3] Checking dependencies...
python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -q opencv-python ultralytics tqdm
)

REM Run auto-labeling
echo [3/3] Auto-labeling images with pre-trained YOLO...
echo.
echo This will use yolo11m.pt to detect objects
echo Default confidence threshold: 0.5
echo.

python YoloLabel\auto_label.py ^
    --images dataset\images\train ^
    --output dataset\labels\train ^
    --model yolo11m.pt ^
    --conf 0.5 ^
    --visualize ^
    --viz-limit 50

if errorlevel 1 (
    echo.
    echo ERROR: Auto-labeling failed!
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================
echo   Auto-Labeling Complete!
echo ======================================
echo.
echo ✅ Results:
echo   - Labels saved to: dataset\labels\train\
echo   - Visualizations: dataset\visualizations\
echo.
echo 📋 Next steps:
echo.
echo 1. Review auto-generated labels:
echo    - Check visualizations folder for sample images with bboxes
echo    - Open dataset\labels\train\ to review .txt files
echo.
echo 2. Use Label Studio to verify and correct:
echo    - Run: 3.label.bat
echo    - Import generated labels
echo    - Fix any false positives/negatives
echo.
echo 3. After correction, start training:
echo    - Run: 4.train.bat
echo.
echo ======================================
echo.
pause
