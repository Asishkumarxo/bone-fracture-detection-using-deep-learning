@echo off
cd /d "%~dp0"
echo Starting BoneSight Streamlit Frontend on http://localhost:8501 ...
call .venv\Scripts\activate.bat
streamlit run frontend/app.py --server.port 8501
pause

