================================================================================
                    STRATOSPHERE TRADING AI
                    XAUUSD HFT Scalping System
================================================================================

QUICK START:
------------
1. Double-click QUICK_START.bat
   - Installs everything automatically
   - Downloads gold data
   - Runs a backtest to verify

FULL MENU:
----------
1. Double-click RUN_TRADING_AI.bat
   - Choose from menu options
   - Web dashboard, backtesting, live trading

MANUAL COMMANDS:
----------------
# Install packages
pip install -r requirements_hft.txt

# Run backtest
python stratosphere_bot.py --backtest --start 2024-01-01

# Run live trading (requires MT5)
python stratosphere_bot.py --risk 0.02

# Run web dashboard
python app.py
Then open: http://localhost:5000

# Run gold trading script
python trade_gold.py

FILES:
------
- stratosphere_bot.py  : Main HFT bot with VOMS + Threaded ML
- xauusd_hft_bot.py    : Alternative HFT bot
- app.py               : Web dashboard
- trade_gold.py        : Simple gold trading script
- models/              : ML model files

REQUIREMENTS:
-------------
- Python 3.10+
- MetaTrader5 (for live trading)
- Windows OS (for MT5)

RISK WARNING:
-------------
Trading involves substantial risk. Past performance does not guarantee
future results. Only trade with money you can afford to lose.
Test on demo account first!

================================================================================
