@echo off
setlocal
if exist .env (
  for /f "usebackq tokens=1,2 delims==#" %%a in (".env") do (
    if NOT "%%a"=="" set "%%a=%%b"
  )
)
python -m streamlit run app/ui_app.py
pause
