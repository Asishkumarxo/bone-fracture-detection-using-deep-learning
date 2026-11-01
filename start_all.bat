@echo off
cd /d "%~dp0"
echo ========================================================
echo   Launching Bone Fracture Deep Learning System
echo ========================================================
echo.

echo [1/2] Starting FastAPI Backend on Port 8000...
start "BoneSight Backend (Port 8000)" cmd /k "call .venv\Scripts\activate.bat && uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo Waiting for backend models to initialize...
timeout /t 4 /nobreak >nul

echo [2/2] Starting Streamlit Frontend on Port 8501...
start "BoneSight Frontend (Port 8501)" cmd /k "call .venv\Scripts\activate.bat && streamlit run frontend/app.py --server.port 8501"

echo.
echo ========================================================
echo   BoneSight Services Successfully Launched!
echo ========================================================
echo   Frontend Web App URL : http://localhost:8501
echo   Backend REST API URL : http://localhost:8000
echo   Interactive API Docs : http://localhost:8000/docs
echo ========================================================
echo   Keep the two opened command windows running.
echo   To stop everything, simply close those two windows.
echo ========================================================
pause

