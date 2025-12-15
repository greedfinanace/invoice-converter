@echo off
echo Starting BTC HFT Bot...
cd /d "%~dp0"
python models/btc_hft_predictor.py
pause
