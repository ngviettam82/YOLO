@echo off
REM Start Image Server for Label Studio
REM This script starts the HTTP image server that serves images to Label Studio

echo.
echo ============================================================================
echo   Image Server for Label Studio
echo ============================================================================
echo.

REM Navigate to project root
cd /d "%~dp0.."

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Find the most recent label_studio_* project folder
for /d %%D in (label_studio_*) do (
    set "PROJECT_DIR=%%D"
)

if not defined PROJECT_DIR (
    echo [ERROR] No Label Studio project found!
    echo Please run: import_to_label_studio.bat first
    pause
    exit /b 1
)

echo [INFO] Using project: %PROJECT_DIR%
echo.

REM Check if serve_images.py exists
if not exist "%PROJECT_DIR%\serve_images.py" (
    echo [ERROR] serve_images.py not found in %PROJECT_DIR%
    echo Please run: import_to_label_studio.bat first
    pause
    exit /b 1
)

echo [INFO] Starting image server...
echo.

cd "%PROJECT_DIR%"
python serve_images.py

REM Deactivate on exit
deactivate

pause
