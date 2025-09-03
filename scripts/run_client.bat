@echo off
echo ========================================
echo Starting ETPA Federated Client...
echo ========================================

call .venv\Scripts\activate

REM Replace with your actual client script path
python client/main.py

pause
