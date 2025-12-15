"""
Gold (XAUUSD) Trading AI
========================
Run: python trade_gold.py
"""
from models import XGBoostPredictor

def main():
    print("=" * 50)
    print("🥇 GOLD (XAUUSD) TRADING AI")
    print("=" * 50)
    
    # Initialize and run everything in one call
    predictor = XGBoostPredictor(look_back=20)
    results = predictor.trade_gold(period="2y", initial_capital=10000)
    
    # Display results
    pred = results["prediction"]
    bt = results["backtest"]
    
    print(f"\n{'=' * 50}")
    print("📊 RESULTS")
    print(f"{'=' * 50}")
    print(f"Current Price:   ${pred['latest_price']}")
    print(f"Signal:          {pred['signal']}")
    print(f"Confidence:      {pred['confidence']}%")
    print(f"\nBacktest Return: {bt['return_pct']}%")
    print(f"Final Value:     ${bt['final_value']:,.2f}")
    print(f"Total Trades:    {bt['total_trades']}")
    
    # Save model
    predictor.save("gold_model.pkl")
    print("\n✅ Model saved to gold_model.pkl")

if __name__ == "__main__":
    main()
