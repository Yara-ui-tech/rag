@echo off
TITLE RAG Training Studio - Production WSGI Server
echo =======================================================================
echo         RAG TRAINING STUDIO - PRODUCTION WSGI LAUNCHER
echo =======================================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

:: Install/verify dependencies
echo [*] Checking dependencies...
python -m pip install -q -r requirements.txt

:: Start production WSGI server
echo [*] Launching Production Server on http://localhost:5050 ...
echo.
python server.py

pause
