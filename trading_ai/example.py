"""
Trading AI - Example Usage
==========================
Run: python example.py
"""
from data import DataFetcher
from indicators import add_indicators
from models import PricePredictor
from strategies import MLStrategy
from backtest import Backtester

def main():
    # 1. Fetch data (pick one)
    fetcher = DataFetcher()
    
    print("Fetching stock data for AAPL...")
    df = fetcher.fetch("AAPL", "stock", period="2y")
    
    # For crypto: df = fetcher.fetch("BTC/USDT", "crypto", limit=730)
    # For forex:  df = fetcher.fetch("EURUSD", "forex", period="2y")
    
    print(f"Got {len(df)} rows of data\n")
    
    # 2. Add technical indicators
    df = add_indicators(df)
    print(f"Added indicators, {len(df)} rows after cleanup\n")
    
    # 3. Train ML model
    print("Training ML model...")
    predictor = PricePredictor(model_type="rf")
    metrics = predictor.train(df, target_days=5)
    print(f"Train accuracy: {metrics['train_accuracy']:.2%}")
    print(f"Test accuracy:  {metrics['test_accuracy']:.2%}\n")
    
    # 4. Create strategy and backtest
    strategy = MLStrategy(predictor, buy_threshold=0.6, sell_threshold=0.4)
    backtester = Backtester(initial_capital=10000)
    
    print("Running backtest...")
    results = backtester.run(df, strategy)
    
    print(f"\n=== BACKTEST RESULTS ===")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value:     ${results['final_value']:,.2f}")
    print(f"Return:          {results['return_pct']}%")
    print(f"Total Trades:    {results['total_trades']}")
    
    # 5. Save model for later use
    predictor.save("trained_model.pkl")
    print("\nModel saved to trained_model.pkl")

if __name__ == "__main__":
    main()
