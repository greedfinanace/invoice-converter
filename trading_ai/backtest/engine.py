"""Simple backtesting engine."""
import pandas as pd
from ..strategies.base import Strategy, SignalType

class Backtester:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
    
    def run(self, df: pd.DataFrame, strategy: Strategy) -> dict:
        """Run backtest on historical data."""
        capital = self.initial_capital
        position = 0
        trades = []
        
        for i in range(50, len(df)):  # Start after indicators warm up
            current_data = df.iloc[:i+1]
            signal = strategy.generate_signal(current_data)
            price = df.iloc[i]["close"]
            
            if signal == SignalType.BUY and position == 0:
                position = capital / price
                capital = 0
                trades.append({"type": "BUY", "price": price, "idx": i})
            
            elif signal == SignalType.SELL and position > 0:
                capital = position * price
                position = 0
                trades.append({"type": "SELL", "price": price, "idx": i})
        
        # Close any open position
        final_value = capital + (position * df.iloc[-1]["close"])
        returns = (final_value - self.initial_capital) / self.initial_capital * 100
        
        return {
            "initial_capital": self.initial_capital,
            "final_value": round(final_value, 2),
            "return_pct": round(returns, 2),
            "total_trades": len(trades),
            "trades": trades
        }
