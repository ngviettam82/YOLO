@echo off
REM ===================================================================
REM Dataset Preparation and Labeling Helper
REM Easy management for YOLO dataset preparation
REM ===================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================
echo YOLO Dataset Management Tool
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

:menu
echo.
echo ========================================
echo Dataset Operations
echo ========================================
echo.
echo 1. Split raw dataset into train/val/test (70/20/10)
echo 2. Split with custom ratios
echo 3. Open annotation tool (LabelImg)
echo 4. Create dataset configuration
echo 5. View dataset information
echo 6. Exit
echo.
set /p choice="Select operation (1-6): "

if "%choice%"=="1" goto split_default
if "%choice%"=="2" goto split_custom
if "%choice%"=="3" goto label
if "%choice%"=="4" goto config
if "%choice%"=="5" goto info
if "%choice%"=="6" goto exit
echo Invalid choice!
goto menu

:split_default
echo.
echo ========================================
echo Splitting dataset (default: 70/20/10)
echo ========================================
echo.
python scripts/split_dataset.py
pause
goto menu

:split_custom
echo.
echo ========================================
echo Custom Dataset Split
echo ========================================
echo.
set /p train="Enter training ratio (0-1, default 0.7): "
if "!train!"=="" set train=0.7

set /p val="Enter validation ratio (0-1, default 0.2): "
if "!val!"=="" set val=0.2

set /p test="Enter test ratio (0-1, default 0.1): "
if "!test!"=="" set test=0.1

set /p move="Move files instead of copy? (y/n, default n): "
if /i "!move!"=="y" (
    python scripts/split_dataset.py --train !train! --val !val! --test !test! --move
) else (
    python scripts/split_dataset.py --train !train! --val !val! --test !test!
)
pause
goto menu

:label
echo.
echo ========================================
echo Image Labeling Tool
echo ========================================
echo.
python scripts/label_images.py
pause
goto menu

:config
echo.
echo ========================================
echo Create Dataset Configuration
echo ========================================
echo.
set /p classes="Enter number of classes (default 1): "
if "!classes!"=="" set classes=1

python scripts/label_images.py --config --num-classes !classes!
pause
goto menu

:info
echo.
echo ========================================
echo Dataset Information
echo ========================================
echo.

set /a train_count=0
set /a val_count=0
set /a test_count=0

for %%f in (dataset\images\train\*) do set /a train_count+=1
for %%f in (dataset\images\val\*) do set /a val_count+=1
for %%f in (dataset\images\test\*) do set /a test_count+=1

echo Train images: !train_count!
echo Val images:   !val_count!
echo Test images:  !test_count!
echo.

set /a train_labels=0
set /a val_labels=0
set /a test_labels=0

for %%f in (dataset\labels\train\*.txt) do set /a train_labels+=1
for %%f in (dataset\labels\val\*.txt) do set /a val_labels+=1
for %%f in (dataset\labels\test\*.txt) do set /a test_labels+=1

echo Train labels: !train_labels!
echo Val labels:   !val_labels!
echo Test labels:  !test_labels!
echo.

if exist "dataset\data.yaml" (
    echo Dataset configuration file found: dataset\data.yaml
) else (
    echo WARNING: No dataset configuration file found
    echo Run 'Create Dataset Configuration' to generate one
)
echo.
pause
goto menu

:exit
echo.
echo Thank you for using YOLO Dataset Manager!
echo.
exit /b 0
