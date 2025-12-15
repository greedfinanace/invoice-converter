"""Flask web app for Trading AI dashboard - Self-contained version."""
from flask import Flask, render_template, request, jsonify
import json
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ta
import warnings
warnings.filterwarnings('ignore')

try:
    import ccxt
    HAS_CCXT = True
except:
    HAS_CCXT = False

app = Flask(__name__)

# ==================== DATA FETCHER ====================

COMMODITY_MAP = {
    "GOLD": "GC=F", "XAUUSD": "GC=F", "XAU/USD": "GC=F",
    "SILVER": "SI=F", "XAGUSD": "SI=F", "XAG/USD": "SI=F",
    "OIL": "CL=F", "PLATINUM": "PL=F", "COPPER": "HG=F"
}

def fetch_data(symbol, asset_type, period="1y", limit=365):
    if asset_type == "stock":
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    
    elif asset_type == "crypto":
        if not HAS_CCXT:
            raise Exception("ccxt not installed for crypto")
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, "1d", limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    
    elif asset_type == "forex":
        if "=X" not in symbol:
            symbol = f"{symbol}=X"
        return fetch_data(symbol, "stock", period)
    
    elif asset_type == "commodity":
        ticker = COMMODITY_MAP.get(symbol.upper(), symbol)
        return fetch_data(ticker, "stock", period)
    
    raise ValueError(f"Unknown asset type: {asset_type}")

# ==================== INDICATORS ====================

def add_indicators(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
    
    df['sma_10'] = ta.trend.sma_indicator(close, window=10)
    df['sma_20'] = ta.trend.sma_indicator(close, window=20)
    df['sma_50'] = ta.trend.sma_indicator(close, window=50)
    df['ema_12'] = ta.trend.ema_indicator(close, window=12)
    df['macd'] = ta.trend.macd(close)
    df['rsi'] = ta.momentum.rsi(close, window=14)
    df['bb_high'] = ta.volatility.bollinger_hband(close, window=20)
    df['bb_low'] = ta.volatility.bollinger_lband(close, window=20)
    df['atr'] = ta.volatility.average_true_range(high, low, close, window=14)
    df['volume_sma'] = volume.rolling(window=20).mean()
    
    return df.dropna()

# ==================== ROUTES ====================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json
    symbol = data.get("symbol", "AAPL").upper()
    asset_type = data.get("asset_type", "stock")
    
    # Auto-detect asset type for common symbols
    if symbol in COMMODITY_MAP or symbol in ["GOLD", "SILVER", "OIL"]:
        asset_type = "commodity"
    
    try:
        # Fetch data
        df = fetch_data(symbol, asset_type, period="1y", limit=365)
        
        if len(df) < 100:
            return jsonify({"success": False, "error": f"Not enough data for {symbol}. Got {len(df)} rows."})
        df = add_indicators(df)
        
        # Prepare features
        features = ['sma_10', 'sma_20', 'sma_50', 'rsi', 'macd', 'bb_high', 'bb_low', 'volume_sma']
        look_back = 20
        
        X, y = [], []
        for i in range(look_back, len(df) - 1):
            X.append(df[features].iloc[i-look_back:i].values.flatten())
            y.append(1 if df['close'].iloc[i+1] > df['close'].iloc[i] else 0)
        
        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                                   eval_metric='logloss', random_state=42)
        model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
        
        train_acc = float(accuracy_score(y_train, model.predict(X_train_scaled)))
        test_acc = float(accuracy_score(y_test, model.predict(X_test_scaled)))
        
        # Current prediction
        X_latest = df[features].iloc[-look_back:].values.flatten().reshape(1, -1)
        X_latest_scaled = scaler.transform(X_latest)
        pred = int(model.predict(X_latest_scaled)[0])
        prob = float(model.predict_proba(X_latest_scaled)[0][1])
        signal = "BUY" if pred == 1 else "SELL"
        
        # Backtest
        capital, position, trades = 10000, 0, []
        for i in range(look_back + 50, len(df)):
            X_bt = df[features].iloc[i-look_back:i].values.flatten().reshape(1, -1)
            X_bt_scaled = scaler.transform(X_bt)
            bt_pred = model.predict(X_bt_scaled)[0]
            price = float(df['close'].iloc[i])
            
            if bt_pred == 1 and position == 0:
                position = capital / price
                capital = 0
                trades.append({"type": "BUY", "price": price, "idx": i})
            elif bt_pred == 0 and position > 0:
                capital = position * price
                position = 0
                trades.append({"type": "SELL", "price": price, "idx": i})
        
        final_value = capital + (position * float(df['close'].iloc[-1]))
        returns = (final_value - 10000) / 10000 * 100
        
        # Create chart
        chart = create_chart(df, trades)
        
        confidence = float(round(prob * 100 if pred == 1 else (1-prob) * 100, 2))
        latest_price = float(round(float(df['close'].iloc[-1]), 2))
        
        return jsonify({
            "success": True,
            "metrics": {"train_accuracy": train_acc, "test_accuracy": test_acc},
            "backtest": {"initial_capital": 10000, "final_value": float(round(final_value, 2)), 
                        "return_pct": float(round(returns, 2)), "total_trades": len(trades), "trades": trades},
            "signal": signal,
            "confidence": confidence,
            "chart": chart,
            "latest_price": latest_price
        })
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})

def create_chart(df, trades):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05)
    
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                  low=df["low"], close=df["close"], name="Price"), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_high"], line=dict(color="gray", width=1),
                             name="BB Upper", opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_low"], line=dict(color="gray", width=1),
                             name="BB Lower", opacity=0.5, fill="tonexty"), row=1, col=1)
    
    for trade in trades:
        if trade["idx"] < len(df):
            idx = df.index[trade["idx"]]
            color = "green" if trade["type"] == "BUY" else "red"
            symbol = "triangle-up" if trade["type"] == "BUY" else "triangle-down"
            fig.add_trace(go.Scatter(x=[idx], y=[trade["price"]], mode="markers",
                                     marker=dict(size=12, color=color, symbol=symbol),
                                     showlegend=False), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], line=dict(color="purple"), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], line=dict(color="blue"), name="MACD"), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    fig.update_layout(height=700, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
    return json.loads(plotly.io.to_json(fig))

if __name__ == "__main__":
    print("🚀 Starting Trading AI Dashboard...")
    print("📊 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, host="0.0.0.0")
