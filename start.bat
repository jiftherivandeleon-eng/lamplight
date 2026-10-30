@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install it from python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
where ollama >nul 2>nul
if errorlevel 1 echo Note: Ollama was not found. Install it from ollama.com or the chat will not answer.
if not exist .venv\ready (
  echo First start: setting up. This needs internet and takes a few minutes.
  python -m venv .venv
  if errorlevel 1 goto fail
  .venv\Scripts\python -m pip install -r requirements.txt
  if errorlevel 1 goto fail
  echo ok> .venv\ready
)
.venv\Scripts\python server.py
pause
exit /b 0
:fail
echo Setup failed. Check your internet connection and run start.bat again.
pause
exit /b 1
