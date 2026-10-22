@echo off
REM Daily stock agent run (for Windows Task Scheduler).
REM 1. Predicts with today's closes (run it late evening, after all markets closed).
REM 2. Scores older predictions against what really happened.
REM Everything is appended to predictions\daily_log.txt

cd /d "C:\stock predictor"
if not exist predictions mkdir predictions

REM Full path to Python, because the scheduler may not know the short "python" command.
set PY=C:\Users\user\AppData\Local\Python\pythoncore-3.14-64\python.exe
if not exist "%PY%" set PY=python

echo. >> predictions\daily_log.txt
echo ===== %date% %time% ===== >> predictions\daily_log.txt
REM Refresh company results: new earnings reports (only stocks that just reported)
REM and the US SEC filings (needs the SEC_EMAIL variable, set once with setx).
"%PY%" download_earnings.py --due >> predictions\daily_log.txt 2>&1
"%PY%" download_fundamentals.py --refresh >> predictions\daily_log.txt 2>&1

REM 5-day model (downloads today's prices)
"%PY%" predict_today.py --include-today >> predictions\daily_log.txt 2>&1
REM 20-day RANKING main model (retrained monthly by run_monthly.bat; reuses the prices just downloaded)
set MAIN=lightgbm_model_h20_rank_main.txt
if not exist "%MAIN%" set MAIN=lightgbm_model_h20_rank_v3.txt
"%PY%" predict_today.py --include-today --skip-download --model %MAIN% >> predictions\daily_log.txt 2>&1
REM Notify about tracked stocks whose ranking changed (web interface + Windows pop-up)
"%PY%" app\notify.py >> predictions\daily_log.txt 2>&1
REM Score both models' past predictions
"%PY%" predict_today.py --evaluate >> predictions\daily_log.txt 2>&1
"%PY%" predict_today.py --evaluate --model lightgbm_model_h20.txt >> predictions\daily_log.txt 2>&1
"%PY%" predict_today.py --evaluate --model lightgbm_model_h20_rank.txt >> predictions\daily_log.txt 2>&1
"%PY%" predict_today.py --evaluate --model lightgbm_model_h20_rank_v3.txt >> predictions\daily_log.txt 2>&1
if exist lightgbm_model_h20_rank_main.txt "%PY%" predict_today.py --evaluate --model lightgbm_model_h20_rank_main.txt >> predictions\daily_log.txt 2>&1
