@echo off
title Insurance Fraud Detection Engine
color 0A
cls

echo.
echo  ============================================================
echo   INSURANCE FRAUD CLAIMS DETECTION ENGINE
echo   FastAPI + XGBoost + SHAP   ^|   http://localhost:8000
echo  ============================================================
echo.

cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python 3.10+
    pause & exit /b 1
)

if not exist "data\insurance_claims.csv" (
    echo  [ERROR] Dataset missing: data\insurance_claims.csv
    pause & exit /b 1
)

python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo  [INFO] Installing web dependencies...
    pip install -r requirements_web.txt -q
)

echo  [OK] Dataset found
echo  [OK] Dependencies ready
echo.
echo  Starting server — models train on first load (~30s)...
echo  Open http://localhost:8000 in your browser
echo  Press Ctrl+C to stop
echo.
echo  ============================================================
echo.

start "" http://localhost:8000
uvicorn web.main:app --host 0.0.0.0 --port 8000 --reload

pause
