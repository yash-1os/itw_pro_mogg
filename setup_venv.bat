@echo off
cd /d "%~dp0"

if exist venv\Scripts\python.exe (
    echo Venv already exists, skipping creation.
) else (
    echo Creating virtual environment...
    py -3.12 -m venv venv
    if errorlevel 1 (
        echo Failed to create venv. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

echo Installing requirements...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo Setup complete. Now run Start_App.bat
pause
