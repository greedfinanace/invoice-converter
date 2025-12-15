"""Technical indicators using ta library."""
import pandas as pd
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to dataframe."""
    df = df.copy()
    
    # Trend
    df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
    df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
    df["macd"] = ta.trend.macd_diff(df["close"])
    
    # Momentum
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["stoch"] = ta.momentum.stoch(df["high"], df["low"], df["close"])
    
    # Volatility
    df["bb_high"] = ta.volatility.bollinger_hband(df["close"])
    df["bb_low"] = ta.volatility.bollinger_lband(df["close"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
    
    # Volume
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    
    return df.dropna()
