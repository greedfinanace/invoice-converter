"""
Complete XGBoost Trading AI - All-in-One
========================================
Supports: Stocks, Crypto, Forex, Commodities (Gold/Silver)
Features: Data fetching, indicators, ML prediction, backtesting, risk management
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import ta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False


class XGBoostPredictor:
    """Complete trading AI with XGBoost, data fetching, and backtesting."""
    
    # Asset symbol mappings
    COMMODITY_MAP = {
        "GOLD": "GC=F", "XAUUSD": "GC=F", "XAU/USD": "GC=F",
        "SILVER": "SI=F", "XAGUSD": "SI=F", "XAG/USD": "SI=F",
        "OIL": "CL=F", "CRUDE": "CL=F", "WTI": "CL=F",
        "PLATINUM": "PL=F", "COPPER": "HG=F",
        "NATGAS": "NG=F", "WHEAT": "ZW=F", "CORN": "ZC=F"
    }
    
    def __init__(self, look_back: int = 20, model_type: str = "xgboost"):
        self.look_back = look_back
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler()
        self.features = []
        self.crypto_exchange = None
        self.trained = False
        
    # ==================== DATA FETCHING ====================
    
    def fetch_stock(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    
    def fetch_crypto(self, symbol: str, timeframe: str = "1d", limit: int = 730) -> pd.DataFrame:
        """Fetch crypto data from Binance."""
        if not HAS_CCXT:
            raise ImportError("ccxt not installed. Run: pip install ccxt")
        if not self.crypto_exchange:
            self.crypto_exchange = ccxt.binance()
        
        ohlcv = self.crypto_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    
    def fetch_forex(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch forex data."""
        if "=X" not in symbol:
            symbol = f"{symbol}=X"
        return self.fetch_stock(symbol, period)
    
    def fetch_commodity(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch commodity data (Gold, Silver, Oil, etc.)."""
        ticker = self.COMMODITY_MAP.get(symbol.upper(), symbol)
        return self.fetch_stock(ticker, period)
    
    def fetch_data(self, symbol: str, asset_type: str = "stock", period: str = "2y", limit: int = 730) -> pd.DataFrame:
        """Universal data fetcher for any asset type."""
        fetchers = {
            "stock": lambda: self.fetch_stock(symbol, period),
            "crypto": lambda: self.fetch_crypto(symbol, limit=limit),
            "forex": lambda: self.fetch_forex(symbol, period),
            "commodity": lambda: self.fetch_commodity(symbol, period)
        }
        
        if asset_type not in fetchers:
            raise ValueError(f"Unknown asset type: {asset_type}. Use: stock, crypto, forex, commodity")
        
        return fetchers[asset_type]()
    
    # ==================== TECHNICAL INDICATORS ====================
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators."""
        df = df.copy()
        
        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # === TREND INDICATORS ===
        df['SMA_10'] = ta.trend.sma_indicator(close, window=10)
        df['SMA_20'] = ta.trend.sma_indicator(close, window=20)
        df['SMA_50'] = ta.trend.sma_indicator(close, window=50)
        df['EMA_12'] = ta.trend.ema_indicator(close, window=12)
        df['EMA_26'] = ta.trend.ema_indicator(close, window=26)
        df['MACD'] = ta.trend.macd(close)
        df['MACD_Signal'] = ta.trend.macd_signal(close)
        df['MACD_Diff'] = ta.trend.macd_diff(close)
        df['ADX'] = ta.trend.adx(high, low, close, window=14)
        
        # === MOMENTUM INDICATORS ===
        df['RSI'] = ta.momentum.rsi(close, window=14)
        df['RSI_6'] = ta.momentum.rsi(close, window=6)
        df['Stoch_K'] = ta.momentum.stoch(high, low, close, window=14)
        df['Stoch_D'] = ta.momentum.stoch_signal(high, low, close, window=14)
        df['Williams_R'] = ta.momentum.williams_r(high, low, close, window=14)
        df['ROC'] = ta.momentum.roc(close, window=10)
        df['MOM'] = close.diff(10)
        
        # === VOLATILITY INDICATORS ===
        df['BB_High'] = ta.volatility.bollinger_hband(close, window=20)
        df['BB_Low'] = ta.volatility.bollinger_lband(close, window=20)
        df['BB_Mid'] = ta.volatility.bollinger_mavg(close, window=20)
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
        df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)
        df['ATR_Pct'] = df['ATR'] / close * 100
        
        # === VOLUME INDICATORS ===
        df['Volume_SMA'] = volume.rolling(window=20).mean()
        df['Volume_Ratio'] = volume / df['Volume_SMA']
        df['OBV'] = ta.volume.on_balance_volume(close, volume)
        df['MFI'] = ta.volume.money_flow_index(high, low, close, volume, window=14)
        
        # === PRICE PATTERNS ===
        df['Price_Change'] = close.pct_change() * 100
        df['Price_Change_5'] = close.pct_change(5) * 100
        df['High_Low_Pct'] = (high - low) / close * 100
        df['Close_Position'] = (close - low) / (high - low)  # Where close is in day's range
        
        # === SUPPORT/RESISTANCE ===
        df['Resistance_20'] = high.rolling(window=20).max()
        df['Support_20'] = low.rolling(window=20).min()
        df['Price_vs_Resistance'] = close / df['Resistance_20']
        df['Price_vs_Support'] = close / df['Support_20']
        
        # === TREND STRENGTH ===
        df['Trend_SMA'] = (df['SMA_10'] > df['SMA_20']).astype(int)
        df['Above_SMA_50'] = (close > df['SMA_50']).astype(int)
        
        # Store feature columns
        self.features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']]
        
        # Rename for consistency
        df['Close'] = close
        df['Volume'] = volume
        
        return df.dropna()
    
    # ==================== MODEL TRAINING ====================
    
    def _create_model(self):
        """Create the ML model based on type."""
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=20
            )
        elif self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
    
    def prepare_data(self, df: pd.DataFrame, target_days: int = 1) -> tuple:
        """Prepare sequences for training with configurable target."""
        X, y = [], []
        
        for i in range(self.look_back, len(df) - target_days):
            X.append(df[self.features].iloc[i-self.look_back:i].values.flatten())
            # Target: 1 if price goes up in next N days
            future_price = df['Close'].iloc[i + target_days]
            current_price = df['Close'].iloc[i]
            y.append(1 if future_price > current_price else 0)
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame, target_days: int = 1, test_size: float = 0.2) -> dict:
        """Train the model with comprehensive metrics."""
        df = self.add_technical_indicators(df)
        X, y = self.prepare_data(df, target_days)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = self._create_model()
        
        if self.model_type == "xgboost":
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        self.trained = True
        
        return {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "features_used": len(self.features),
            "target_days": target_days
        }
    
    def train_on_symbol(self, symbol: str, asset_type: str = "stock", **kwargs) -> dict:
        """Fetch data and train in one step."""
        df = self.fetch_data(symbol, asset_type)
        return self.train(df, **kwargs)
    
    # ==================== PREDICTION ====================
    
    def predict(self, df: pd.DataFrame) -> int:
        """Predict signal: 1 (BUY) or 0 (SELL)."""
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        df = self.add_technical_indicators(df)
        
        if len(df) < self.look_back:
            return -1  # Not enough data
        
        X = df[self.features].iloc[-self.look_back:].values.flatten().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        return int(self.model.predict(X_scaled)[0])
    
    def predict_proba(self, df: pd.DataFrame) -> float:
        """Get probability of upward movement."""
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        df = self.add_technical_indicators(df)
        
        if len(df) < self.look_back:
            return 0.5
        
        X = df[self.features].iloc[-self.look_back:].values.flatten().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        return float(self.model.predict_proba(X_scaled)[0][1])
    
    def predict_symbol(self, symbol: str, asset_type: str = "stock") -> dict:
        """Get complete prediction for a symbol."""
        df = self.fetch_data(symbol, asset_type)
        signal = self.predict(df)
        prob = self.predict_proba(df)
        
        close_col = 'close' if 'close' in df.columns else 'Close'
        latest_price = float(df[close_col].iloc[-1])
        
        return {
            "symbol": symbol,
            "asset_type": asset_type,
            "signal": "BUY" if signal == 1 else "SELL",
            "confidence": round(prob * 100, 2) if signal == 1 else round((1 - prob) * 100, 2),
            "probability": round(prob, 4),
            "latest_price": round(latest_price, 2)
        }
    
    # ==================== BACKTESTING ====================
    
    def run_backtest(self, df: pd.DataFrame, initial_capital: float = 10000, 
                     stop_loss: float = None, take_profit: float = None) -> dict:
        """Run backtest with optional stop-loss and take-profit."""
        df_ind = self.add_technical_indicators(df)
        
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_capital]
        
        close_col = 'Close' if 'Close' in df_ind.columns else 'close'
        
        for i in range(self.look_back + 50, len(df_ind)):
            X = df_ind[self.features].iloc[i-self.look_back:i].values.flatten().reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            
            price = float(df_ind[close_col].iloc[i])
            
            # Check stop-loss / take-profit
            if position > 0 and entry_price > 0:
                pnl_pct = (price - entry_price) / entry_price
                
                if stop_loss and pnl_pct <= -stop_loss:
                    capital = position * price
                    trades.append({"type": "STOP_LOSS", "price": price, "idx": i, "pnl_pct": round(pnl_pct * 100, 2)})
                    position = 0
                    entry_price = 0
                elif take_profit and pnl_pct >= take_profit:
                    capital = position * price
                    trades.append({"type": "TAKE_PROFIT", "price": price, "idx": i, "pnl_pct": round(pnl_pct * 100, 2)})
                    position = 0
                    entry_price = 0
            
            # Regular signals
            if pred == 1 and position == 0:  # BUY
                position = capital / price
                entry_price = price
                capital = 0
                trades.append({"type": "BUY", "price": price, "idx": i})
            elif pred == 0 and position > 0:  # SELL
                pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0
                capital = position * price
                trades.append({"type": "SELL", "price": price, "idx": i, "pnl_pct": round(pnl_pct * 100, 2)})
                position = 0
                entry_price = 0
            
            # Track equity
            current_equity = capital + (position * price)
            equity_curve.append(current_equity)
        
        final_value = capital + (position * float(df_ind[close_col].iloc[-1]))
        returns = (final_value - initial_capital) / initial_capital * 100
        
        # Calculate metrics
        winning_trades = [t for t in trades if t.get('pnl_pct', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl_pct', 0) < 0]
        
        return {
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "return_pct": round(returns, 2),
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(len(winning_trades) / max(len(winning_trades) + len(losing_trades), 1) * 100, 2),
            "max_equity": round(max(equity_curve), 2),
            "min_equity": round(min(equity_curve), 2),
            "trades": trades
        }
    
    # ==================== COMPLETE WORKFLOWS ====================
    
    def trade_gold(self, period: str = "2y", initial_capital: float = 10000, 
                   stop_loss: float = 0.02, take_profit: float = 0.05) -> dict:
        """Complete gold (XAUUSD) trading workflow."""
        print("📊 Fetching XAUUSD data...")
        df = self.fetch_data("XAUUSD", "commodity", period=period)
        print(f"Got {len(df)} days of data")
        
        print("\n🤖 Training model...")
        metrics = self.train(df, target_days=1)
        print(f"Train Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"Test Accuracy:  {metrics['test_accuracy']:.2%}")
        
        print("\n📈 Getting prediction...")
        prediction = self.predict_symbol("XAUUSD", "commodity")
        
        print("\n📉 Running backtest...")
        backtest = self.run_backtest(df, initial_capital, stop_loss, take_profit)
        
        return {"metrics": metrics, "prediction": prediction, "backtest": backtest}
    
    def trade_symbol(self, symbol: str, asset_type: str = "stock", period: str = "2y",
                     initial_capital: float = 10000, stop_loss: float = 0.02, 
                     take_profit: float = 0.05) -> dict:
        """Complete trading workflow for any symbol."""
        print(f"📊 Fetching {symbol} data...")
        df = self.fetch_data(symbol, asset_type, period=period)
        print(f"Got {len(df)} days of data")
        
        print("\n🤖 Training model...")
        metrics = self.train(df, target_days=1)
        print(f"Train Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"Test Accuracy:  {metrics['test_accuracy']:.2%}")
        
        print("\n📈 Getting prediction...")
        prediction = self.predict_symbol(symbol, asset_type)
        
        print("\n📉 Running backtest...")
        backtest = self.run_backtest(df, initial_capital, stop_loss, take_profit)
        
        return {"metrics": metrics, "prediction": prediction, "backtest": backtest}
    
    def scan_multiple(self, symbols: list, asset_type: str = "stock") -> list:
        """Scan multiple symbols and return signals."""
        results = []
        for symbol in symbols:
            try:
                df = self.fetch_data(symbol, asset_type, period="1y")
                self.train(df)
                pred = self.predict_symbol(symbol, asset_type)
                results.append(pred)
            except Exception as e:
                results.append({"symbol": symbol, "error": str(e)})
        return results
    
    # ==================== SAVE/LOAD ====================
    
    def save(self, path: str):
        """Save model to file."""
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "look_back": self.look_back,
            "features": self.features,
            "model_type": self.model_type,
            "trained": self.trained
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load(self, path: str):
        """Load model from file."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.look_back = data["look_back"]
        self.features = data["features"]
        self.model_type = data.get("model_type", "xgboost")
        self.trained = data.get("trained", True)
        print(f"✅ Model loaded from {path}")


# ==================== STANDALONE USAGE ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 XGBOOST TRADING AI - STANDALONE TEST")
    print("=" * 60)
    
    predictor = XGBoostPredictor(look_back=20, model_type="xgboost")
    
    # Test with Gold
    results = predictor.trade_gold(period="2y", initial_capital=10000, stop_loss=0.02, take_profit=0.05)
    
    pred = results["prediction"]
    bt = results["backtest"]
    
    print(f"\n{'=' * 60}")
    print("📊 FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"Symbol:          {pred['symbol']}")
    print(f"Current Price:   ${pred['latest_price']}")
    print(f"Signal:          {pred['signal']}")
    print(f"Confidence:      {pred['confidence']}%")
    print(f"\nBacktest Return: {bt['return_pct']}%")
    print(f"Final Value:     ${bt['final_value']:,.2f}")
    print(f"Win Rate:        {bt['win_rate']}%")
    print(f"Total Trades:    {bt['total_trades']}")
    
    predictor.save("trading_model.pkl")
