@echo off
python -m pip install -r requirements.txt
python main.py --tickers AAPL --models lstm gru --start 2018-01-01 --forecast_start 2026-04-28 --forecast_end 2026-06-26
pause
