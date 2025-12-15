import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET = os.getenv("BINANCE_SECRET", "")
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
    
    # Risk Management
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.02))
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 5))
    
    # Supported asset types
    ASSET_TYPES = ["stock", "crypto", "forex"]
