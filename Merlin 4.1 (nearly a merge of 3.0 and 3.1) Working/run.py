import subprocess, sys
subprocess.run([sys.executable, "-m", "streamlit", "run", "app/ui_app.py", "--server.headless=true"], check=False)
