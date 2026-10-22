@echo off
REM Monthly automatic retraining (Windows Task Scheduler, first Saturday of the month).
REM Takes about 3 hours. Keep the PC on and plugged in. Log: predictions\monthly_log.txt
cd /d "C:\stock predictor"
if not exist predictions mkdir predictions
set PY=C:\Users\user\AppData\Local\Python\pythoncore-3.14-64\python.exe
if not exist "%PY%" set PY=python
set PYTHONUNBUFFERED=1
echo. >> predictions\monthly_log.txt
"%PY%" monthly_retrain.py >> predictions\monthly_log.txt 2>&1
