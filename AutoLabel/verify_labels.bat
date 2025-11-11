@echo off
REM Verify Labels - Visual Label Verification Tool
REM Display images with their YOLO labels for verification

echo.
echo ========================================
echo   Verify Labels Tool
echo ========================================
echo.

REM Navigate to project root (parent of AutoLabel folder)
cd /d "%~dp0.."

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the verify labels script
python AutoLabel\scripts\verify_labels.py

REM Deactivate virtual environment
deactivate

pause
