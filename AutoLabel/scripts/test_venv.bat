@echo off
REM Test script to verify batch file works

echo Testing batch file configuration...
echo.

REM Navigate to project root (parent of AutoLabel folder)
cd /d "%~dp0.."

echo Current directory: %cd%
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Expected: %cd%\.venv\Scripts\activate.bat
    pause
    exit /b 1
) else (
    echo SUCCESS: Virtual environment found!
    echo Location: %cd%\.venv
)

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo SUCCESS: Virtual environment activated!
echo Python location:
where python
echo.

echo Everything is working correctly!
pause
