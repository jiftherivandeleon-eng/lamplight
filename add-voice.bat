@echo off
cd /d "%~dp0"
if not exist .venv\ready (
  echo Run start.bat once first.
  pause
  exit /b 1
)
.venv\Scripts\python -m pip install faster-whisper
echo Done. Close the app window and run start.bat again.
pause
