@echo off
REM Politely - one-click launcher for Windows.
REM First run: creates a virtual environment and installs the dependencies (CPU PyTorch, a few minutes).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating the virtual environment...
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv\Scripts\python -m pip install -r requirements.txt
  .venv\Scripts\python -m pip install -e .
)
.venv\Scripts\python -X utf8 run_app.py
pause
