
param([string]$ProjectRoot = "$PSScriptRoot/app")
Set-Location $ProjectRoot
streamlit run ui_app.py
