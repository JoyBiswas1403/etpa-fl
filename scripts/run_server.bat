@echo off
echo ========================================
echo Starting ETPA Federated Server...
echo ========================================

call .venv\Scripts\activate

REM Replace with your actual server script path
python server/main.py

pause
