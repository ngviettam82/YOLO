@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo   YOLO Label Studio - Complete Workflow
echo ============================================================================
echo.

cd /d "%~dp0.."

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
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
for /d %%D in (label_studio_*) do (
    set "PROJECT_DIR=%%D"
)

if not defined PROJECT_DIR (
    echo ERROR: Project folder not found!
    pause
    deactivate
    exit /b 1
)

REM STEP 2: Start image server in separate window
echo ============================================================================
echo STEP 2: Starting Image Server in separate window...
echo ============================================================================
echo.

start "Image Server" cmd /c "cd /d "%CD%" && python AutoLabel\scripts\serve_images.py & pause"

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
