@echo off
cd /d "%~dp0"
echo Starting BoneSight FastAPI Backend on http://localhost:8000 ...
call .venv\Scripts\activate.bat
uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause

