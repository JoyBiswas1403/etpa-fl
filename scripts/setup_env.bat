@echo off
echo ========================================
echo Setting up Python virtual environment...
echo ========================================

REM Create virtual environment
python -m venv .venv

REM Activate virtual environment
call .venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt

echo ========================================
echo Environment setup complete!
echo ========================================
pause
