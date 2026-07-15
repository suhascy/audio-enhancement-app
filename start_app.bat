@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ERROR: Python virtual environment was not found.
    pause
    exit /b 1
)

start "Audio Enhancement Backend" cmd /k ""%PYTHON%" -m uvicorn app:app --host 127.0.0.1 --port 8000"

start "Audio Enhancement Frontend" cmd /k ""%PYTHON%" -m http.server 5501 --directory "%~dp0frontend""

timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5501"

endlocal