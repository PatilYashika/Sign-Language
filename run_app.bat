@echo off
title Indian Sign Language AI Studio
echo ===================================================
echo   STARTING INDIAN SIGN LANGUAGE (ISL) AI STUDIO
echo ===================================================
echo Opening your browser to http://localhost:8501 ...
cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
python -m streamlit run app.py --server.headless=false
pause
