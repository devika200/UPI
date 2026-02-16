@echo off
echo ========================================
echo Starting UPI Fraud Detection Backend
echo ========================================
echo.

REM Check if virtual environment exists
if exist venv\ (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
)

echo.
echo Checking Python version...
python --version

echo.
echo Installing/Updating dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Starting Flask Server on port 5000
echo ========================================
echo.
echo Backend will be available at: http://localhost:5000
echo Health check: http://localhost:5000/health
echo.
echo Press Ctrl+C to stop the server
echo.

python mainapp.py
