@echo off
REM Auto-Labeling Tool Batch Script
REM This script runs the auto-labeling tool with the pretrained YOLO model

echo.
echo ========================================
echo   YOLO Auto-Labeling Tool
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

REM Run the auto-labeling script
python AutoLabel\scripts\auto_label.py

REM Deactivate virtual environment
deactivate

pause
