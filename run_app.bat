@echo off
REM Politely - launches the Streamlit interface (Windows).
REM First run: creates a virtual environment and installs the dependencies (a few minutes).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating the virtual environment...
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
)
.venv\Scripts\python -c "import streamlit, sacrebleu, altair" 2>nul || .venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -X utf8 -m streamlit run app.py
pause
