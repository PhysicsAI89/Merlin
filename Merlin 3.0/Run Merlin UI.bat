@echo off
setlocal
cd /d "%~dp0app"
python -m streamlit run ui_app.py
pause