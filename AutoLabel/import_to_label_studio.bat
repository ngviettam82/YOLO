@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo   YOLO Label Studio - Complete Workflow
echo ============================================================================
echo.

REM Resolve YOLO project root (parent of AutoLabel folder) as absolute path
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
echo Script location: %~dp0
echo Project root: !PROJECT_ROOT!

cd /d "!PROJECT_ROOT!"
echo Working directory: %CD%

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at !PROJECT_ROOT!
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

pip show label-studio >nul 2>&1
if errorlevel 1 (
    echo Installing label-studio...
    pip install label-studio
    echo.
)

REM STEP 1: Import data
echo.
echo ============================================================================
echo STEP 1: Importing auto-labeled data...
echo ============================================================================
echo.
python AutoLabel\scripts\import_to_label_studio.py

if errorlevel 1 (
    echo.
    echo ERROR: Import failed!
    pause
    deactivate
    exit /b 1
)

echo.
echo OK - Data imported!
echo.

REM Find the project directory
for /d %%D in ("!PROJECT_ROOT!\label_studio_*") do (
    set "PROJECT_DIR=%%~fD"
)

if not defined PROJECT_DIR (
    echo.
    echo ERROR: No label_studio_* project directory found!
    echo Expected to find in: !PROJECT_ROOT!
    echo.
    echo Make sure STEP 1 ^(import^) completed successfully.
    pause
    deactivate
    exit /b 1
)

REM STEP 2: Start image server in separate window (must run from project dir so /images/ path resolves)
echo ============================================================================
echo STEP 2: Starting Image Server in separate window...
echo ============================================================================
echo.

start "Image Server" cmd /c "cd /d "!PROJECT_DIR!" && python serve_images.py & pause"

echo Image server started in separate window!
echo Waiting 3 seconds...
timeout /t 3 /nobreak

echo.
echo ============================================================================
echo STEP 3: Starting Label Studio
echo ============================================================================
echo.
echo Opening Label Studio at http://localhost:8080
echo (This will open in your browser automatically)
echo.

label-studio

REM STEP 4: Finish
echo.
echo ============================================================================
echo STEP 4: Complete
echo ============================================================================
echo.
echo Label Studio has closed.
echo.
echo Next steps:
echo 1. Export your corrected labels from Label Studio
echo 2. Use exported file for training
echo.
echo (Image server window is still open - close it manually if needed)
echo.

deactivate

pause
exit /b 0
