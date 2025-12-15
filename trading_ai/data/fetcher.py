"""Data fetching for multiple asset types."""
import pandas as pd
import yfinance as yf
import ccxt

class DataFetcher:
    def __init__(self):
        self.crypto_exchange = None
    
    def fetch_stock(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    
    def fetch_crypto(self, symbol: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """Fetch crypto data from Binance."""
        if not self.crypto_exchange:
            self.crypto_exchange = ccxt.binance()
        
        ohlcv = self.crypto_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    
    def fetch_forex(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch forex data (using Yahoo Finance for simplicity)."""
        # Yahoo Finance uses format like 'EURUSD=X'
        if "=X" not in symbol:
            symbol = f"{symbol}=X"
        return self.fetch_stock(symbol, period)
    
    def fetch_commodity(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch commodity data (Gold, Silver, Oil, etc.)."""
        # Yahoo Finance commodity/forex symbols
        commodity_map = {
            "GOLD": "GC=F",
            "XAUUSD": "GC=F",
            "XAU/USD": "GC=F",
            "SILVER": "SI=F",
            "XAGUSD": "SI=F",
            "XAG/USD": "SI=F",
            "OIL": "CL=F",
            "PLATINUM": "PL=F",
            "COPPER": "HG=F"
        }
        ticker = commodity_map.get(symbol.upper(), symbol)
        return self.fetch_stock(ticker, period)
    
    def fetch(self, symbol: str, asset_type: str, **kwargs) -> pd.DataFrame:
        """Universal fetch method."""
        fetchers = {
            "stock": self.fetch_stock,
            "crypto": self.fetch_crypto,
            "forex": self.fetch_forex,
            "commodity": self.fetch_commodity
        }
        return fetchers[asset_type](symbol, **kwargs)
