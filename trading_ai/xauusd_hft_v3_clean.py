"""
XAUUSD HFT Trading System V3 - CLEAN VERSION
=============================================
Simplified, focused trading bot for XAUUSD
- TCN + LightGBM + Microstructure ensemble
- Virtual Order Management (VOM) for SL/TP
- Dynamic lot sizing based on confidence
- Trailing stop for profit protection
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import threading
import time
import pickle
import logging
import csv
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import deque
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ML imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, BatchNormalization, Add, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Suppress TF warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU enabled: {len(gpus)} GPU(s)")


# =============================================================================
# CONFIGURATION - SIMPLE AND CLEAN
# =============================================================================

@dataclass
class Config:
    """Trading configuration - all settings in one place"""
    
    # Symbol
    SYMBOL: str = "XAUUSDm"
    SYMBOL_ALTERNATIVES: Tuple[str, ...] = ("XAUUSDm", "XAUUSD", "GOLD", "GOLDm")
    TIMEFRAME: int = 1  # M1
    POINT_VALUE: float = 0.01  # 1 pip = $0.01 for gold
    
    # Trading thresholds
    MIN_CONFIDENCE: float = 0.35  # Minimum confidence to trade (higher = fewer but better trades)
    
    # TP/SL Settings (in pips)
    TP_PIPS: float = 150.0  # Take profit
    SL_PIPS: float = 80.0   # Stop loss
    MAX_TP_PIPS: float = 700.0  # Cap
    MAX_SL_PIPS: float = 150.0  # Cap
    
    # Trailing stop
    TRAIL_TRIGGER_PIPS: float = 30.0  # Activate after 30 pips profit
    TRAIL_DISTANCE_PIPS: float = 15.0  # Trail 15 pips behind
    
    # Position sizing
    MIN_LOT: float = 0.01
    MAX_LOT: float = 10.0
    
    # Position limits
    MAX_POSITIONS: int = 5
    
    # ML settings
    SEQUENCE_LENGTH: int = 28
    TCN_WEIGHT: float = 0.40
    LGBM_WEIGHT: float = 0.30
    MICRO_WEIGHT: float = 0.30
    TRAINING_CANDLES: int = 10000  # More data = better generalization
    
    # Files
    LOG_FILE: str = "xauusd_hft_v3.log"
    TRADES_CSV: str = "xauusd_trades_v3.csv"
    MODELS_DIR: str = "hft_models_v3"


class TradingSession(Enum):
    ASIAN = "asian"
    LONDON = "london"
    NY = "ny"
    OFF = "off"


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(config: Config) -> logging.Logger:
    logger = logging.getLogger("HFT_V3")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    ch = logging.StreamHandler()
    
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =============================================================================
# VIRTUAL ORDER MANAGEMENT (VOM)
# =============================================================================

@dataclass
class Position:
    """Virtual position tracked in memory"""
    ticket: int
    order_type: int  # 0=BUY, 1=SELL
    volume: float
    entry_price: float
    entry_time: datetime
    sl_price: float
    tp_price: float
    confidence: float
    trailing_active: bool = False
    trailing_sl: Optional[float] = None
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None


class VOM:
    """Virtual Order Management - tracks SL/TP in memory"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.positions: Dict[int, Position] = {}
        self.lock = threading.Lock()
    
    def add(self, pos: Position):
        with self.lock:
            self.positions[pos.ticket] = pos
            self.logger.info(f"VOM ADD: {pos.ticket} | {'BUY' if pos.order_type==0 else 'SELL'} | Entry: {pos.entry_price:.2f} | SL: {pos.sl_price:.2f} | TP: {pos.tp_price:.2f}")
    
    def remove(self, ticket: int) -> Optional[Position]:
        with self.lock:
            return self.positions.pop(ticket, None)
    
    def get_all(self) -> List[Position]:
        with self.lock:
            return list(self.positions.values())
    
    def count(self) -> int:
        with self.lock:
            return len(self.positions)
    
    def check_exit(self, ticket: int, price: float) -> Tuple[bool, str]:
        """Check if position should exit"""
        with self.lock:
            pos = self.positions.get(ticket)
            if not pos:
                return False, ""
            
            pip = self.config.POINT_VALUE
            
            # Calculate profit
            if pos.order_type == 0:  # BUY
                profit_pips = (price - pos.entry_price) / pip
            else:  # SELL
                profit_pips = (pos.entry_price - price) / pip
            
            # Update trailing stop
            if profit_pips >= self.config.TRAIL_TRIGGER_PIPS:
                if not pos.trailing_active:
                    pos.trailing_active = True
                    if pos.order_type == 0:
                        pos.highest_price = price
                        pos.trailing_sl = price - (self.config.TRAIL_DISTANCE_PIPS * pip)
                    else:
                        pos.lowest_price = price
                        pos.trailing_sl = price + (self.config.TRAIL_DISTANCE_PIPS * pip)
                    self.logger.info(f"TRAIL ON: {ticket} at {profit_pips:.1f} pips")
                else:
                    # Update trailing
                    if pos.order_type == 0 and price > pos.highest_price:
                        pos.highest_price = price
                        new_sl = price - (self.config.TRAIL_DISTANCE_PIPS * pip)
                        if new_sl > pos.trailing_sl:
                            pos.trailing_sl = new_sl
                    elif pos.order_type == 1 and price < pos.lowest_price:
                        pos.lowest_price = price
                        new_sl = price + (self.config.TRAIL_DISTANCE_PIPS * pip)
                        if new_sl < pos.trailing_sl:
                            pos.trailing_sl = new_sl
            
            # Check exits
            if pos.order_type == 0:  # BUY
                if price >= pos.tp_price:
                    return True, "TP"
                if pos.trailing_active and pos.trailing_sl and price <= pos.trailing_sl:
                    return True, "TRAIL"
                if price <= pos.sl_price:
                    return True, "SL"
            else:  # SELL
                if price <= pos.tp_price:
                    return True, "TP"
                if pos.trailing_active and pos.trailing_sl and price >= pos.trailing_sl:
                    return True, "TRAIL"
                if price >= pos.sl_price:
                    return True, "SL"
            
            return False, ""


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class Indicators:
    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    @staticmethod
    def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period, min_periods=1).mean().values
        avg_loss = pd.Series(loss).rolling(period, min_periods=1).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss != 0)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        return pd.Series(tr).rolling(period, min_periods=1).mean().values
    
    @staticmethod
    def macd(close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ema12 = Indicators.ema(close, 12)
        ema26 = Indicators.ema(close, 26)
        macd_line = ema12 - ema26
        signal = Indicators.ema(macd_line, 9)
        return macd_line, signal, macd_line - signal
    
    @staticmethod
    def bollinger(close: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sma = pd.Series(close).rolling(period, min_periods=1).mean().values
        std = pd.Series(close).rolling(period, min_periods=1).std().values
        return sma + 2*std, sma, sma - 2*std


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngine:
    def __init__(self, config: Config):
        self.config = config
        self.scaler = RobustScaler()
        self.fitted = False
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        f = pd.DataFrame(index=df.index)
        
        # Price
        f['returns'] = np.diff(close, prepend=close[0]) / np.where(close > 0, close, 1)
        
        # EMAs
        f['ema_5'] = Indicators.ema(close, 5)
        f['ema_10'] = Indicators.ema(close, 10)
        f['ema_20'] = Indicators.ema(close, 20)
        f['ema_cross'] = f['ema_5'] - f['ema_20']
        
        # RSI
        f['rsi_7'] = Indicators.rsi(close, 7)
        f['rsi_14'] = Indicators.rsi(close, 14)
        
        # MACD
        macd, signal, hist = Indicators.macd(close)
        f['macd'] = macd
        f['macd_hist'] = hist
        
        # Bollinger
        bb_up, bb_mid, bb_low = Indicators.bollinger(close)
        f['bb_pos'] = (close - bb_low) / np.where((bb_up - bb_low) > 0, bb_up - bb_low, 1)
        
        # ATR
        f['atr'] = Indicators.atr(high, low, close, 14)
        
        # Momentum
        f['mom_3'] = close - np.roll(close, 3)
        f['mom_5'] = close - np.roll(close, 5)
        
        # Candle patterns
        body = np.abs(close - df['open'].values)
        range_ = high - low
        f['body_ratio'] = body / np.where(range_ > 0, range_, 1)
        f['upper_wick'] = (high - np.maximum(close, df['open'].values)) / np.where(range_ > 0, range_, 1)
        f['lower_wick'] = (np.minimum(close, df['open'].values) - low) / np.where(range_ > 0, range_, 1)
        
        # Clean
        f = f.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return f
    
    def prepare_sequences(self, features: pd.DataFrame, labels: np.ndarray = None):
        """Prepare sequences for TCN"""
        seq_len = self.config.SEQUENCE_LENGTH
        data = features.values
        
        if not self.fitted:
            data = self.scaler.fit_transform(data)
            self.fitted = True
        else:
            data = self.scaler.transform(data)
        
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i])
            if labels is not None:
                y.append(labels[i])
        
        return np.array(X), np.array(y) if labels is not None else None
    
    def create_labels(self, close: np.ndarray, lookahead: int = 5, threshold: float = 3.0) -> np.ndarray:
        """Create labels: 0=SELL, 1=HOLD, 2=BUY"""
        pip = self.config.POINT_VALUE
        labels = np.ones(len(close), dtype=int)
        
        for i in range(len(close) - lookahead):
            future_max = np.max(close[i+1:i+lookahead+1])
            future_min = np.min(close[i+1:i+lookahead+1])
            up = (future_max - close[i]) / pip
            down = (close[i] - future_min) / pip
            
            if up >= threshold and up > down:
                labels[i] = 2
            elif down >= threshold and down > up:
                labels[i] = 0
        
        return labels


# =============================================================================
# ML MODELS
# =============================================================================

class TCNModel:
    """
    Temporal Convolutional Network with ANTI-OVERFITTING measures:
    1. Dropout layers (0.3)
    2. L2 regularization
    3. Early stopping with patience
    4. Validation split (20%)
    5. Reduced model complexity
    6. Data augmentation via noise
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
    
    def build(self, input_shape):
        from tensorflow.keras.regularizers import l2
        
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Smaller model to prevent overfitting
        for filters, dilation in [(32, 1), (64, 2), (64, 4)]:
            conv = Conv1D(filters, 3, dilation_rate=dilation, padding='causal', 
                         activation='relu', kernel_regularizer=l2(0.001))(x)
            conv = BatchNormalization()(conv)
            conv = Dropout(0.3)(conv)  # Higher dropout
            if x.shape[-1] != filters:
                x = Conv1D(filters, 1, padding='same')(x)
            x = Add()([x, conv])
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)  # Smaller dense
        x = Dropout(0.4)(x)  # Higher dropout before output
        outputs = Dense(3, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=Adam(0.0005),  # Lower learning rate
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
    
    def train(self, X, y, epochs=50):
        """
        Train with anti-overfitting:
        - Walk-forward validation (no shuffle)
        - Early stopping with patience=10
        - Reduce LR on plateau
        - Data augmentation with noise
        """
        from tensorflow.keras.callbacks import ReduceLROnPlateau
        
        # Walk-forward split (no shuffle to respect time series)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Add small noise to training data (augmentation)
        noise = np.random.normal(0, 0.01, X_train.shape)
        X_train_aug = X_train + noise
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
        ]
        
        history = self.model.fit(
            X_train_aug, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,  # Smaller batch for better generalization
            callbacks=callbacks,
            verbose=1
        )
        
        # Log final metrics
        val_acc = history.history.get('val_accuracy', [0])[-1]
        val_loss = history.history.get('val_loss', [0])[-1]
        print(f"TCN Training complete: val_acc={val_acc:.3f}, val_loss={val_loss:.3f}")
        
        return history
    
    def predict(self, X):
        if self.model is None:
            return np.array([[0.33, 0.34, 0.33]])
        return self.model.predict(X, verbose=0)
    
    def save(self, path):
        if self.model:
            self.model.save(f"{path}_tcn.keras")
    
    def load(self, path):
        self.model = tf.keras.models.load_model(f"{path}_tcn.keras")


class LGBMModel:
    """
    LightGBM classifier with ANTI-OVERFITTING:
    1. Lower num_leaves (31 instead of 40)
    2. Deeper max_depth limit (5)
    3. L1/L2 regularization (lambda)
    4. Feature fraction (bagging)
    5. Early stopping
    6. Min data in leaf
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
    
    def train(self, X, y):
        # Walk-forward split
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        # Anti-overfitting params
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            
            # Regularization
            'num_leaves': 31,           # Reduced from 40
            'max_depth': 5,             # Reduced from 6
            'min_data_in_leaf': 50,     # Prevent overfitting to small patterns
            'min_gain_to_split': 0.01,  # Require meaningful splits
            
            # L1/L2 regularization
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            
            # Feature/data sampling (bagging)
            'feature_fraction': 0.8,    # Use 80% of features per tree
            'bagging_fraction': 0.8,    # Use 80% of data per tree
            'bagging_freq': 5,
            
            'learning_rate': 0.03,      # Lower LR for better generalization
            'verbose': -1
        }
        
        self.model = lgb.train(
            params, 
            train_data, 
            valid_sets=[val_data],
            num_boost_round=500,  # More rounds with lower LR
            callbacks=[
                lgb.early_stopping(30, verbose=True),
                lgb.log_evaluation(50)
            ]
        )
        
        print(f"LGBM Training complete: best_iteration={self.model.best_iteration}")
    
    def predict(self, X):
        if self.model is None:
            return np.array([[0.33, 0.34, 0.33]])
        return self.model.predict(X)
    
    def save(self, path):
        if self.model:
            self.model.save_model(f"{path}_lgbm.txt")
    
    def load(self, path):
        self.model = lgb.Booster(model_file=f"{path}_lgbm.txt")


class MicroModel:
    """Microstructure pattern detector (rule-based)"""
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Detect short-term patterns. Returns (signal, confidence)"""
        if len(df) < 5:
            return 1, 0.0
        
        recent = df.iloc[-5:]
        close = recent['close'].values
        high = recent['high'].values
        low = recent['low'].values
        open_ = recent['open'].values
        
        score = 0.0
        
        # Pin bar
        body = abs(close[-1] - open_[-1])
        range_ = high[-1] - low[-1]
        if range_ > 0:
            lower_wick = (min(close[-1], open_[-1]) - low[-1]) / range_
            upper_wick = (high[-1] - max(close[-1], open_[-1])) / range_
            if body / range_ < 0.3:
                if lower_wick > 0.5:
                    score += 0.3  # Bullish pin
                elif upper_wick > 0.5:
                    score -= 0.3  # Bearish pin
        
        # Higher high/low
        if high[-1] > high[-2] and low[-1] > low[-2]:
            score += 0.2
        elif low[-1] < low[-2] and high[-1] < high[-2]:
            score -= 0.2
        
        # 3-bar momentum
        mom = (close[-1] - close[-3]) / close[-3] * 100 if close[-3] > 0 else 0
        if mom > 0.02:
            score += 0.15
        elif mom < -0.02:
            score -= 0.15
        
        if score > 0.2:
            return 2, min(abs(score), 0.5)
        elif score < -0.2:
            return 0, min(abs(score), 0.5)
        return 1, 0.0


# =============================================================================
# ENSEMBLE
# =============================================================================

class Ensemble:
    """Combined ML ensemble"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.tcn = TCNModel(config)
        self.lgbm = LGBMModel(config)
        self.micro = MicroModel()
        self.features = FeatureEngine(config)
        self.trained = False
        self.lock = threading.Lock()
    
    def train(self, df: pd.DataFrame):
        """
        Train TCN and LGBM with ANTI-OVERFITTING:
        1. Use more data (7000+ candles)
        2. Walk-forward validation
        3. Label smoothing via threshold adjustment
        4. Class balancing
        """
        self.logger.info("=" * 50)
        self.logger.info("TRAINING ENSEMBLE (Anti-Overfitting Mode)")
        self.logger.info("=" * 50)
        
        with self.lock:
            # Calculate features
            self.logger.info(f"Processing {len(df)} candles...")
            features = self.features.calculate(df)
            
            # Create labels with slightly higher threshold to reduce noise
            labels = self.features.create_labels(df['close'].values, lookahead=5, threshold=4.0)
            
            # Log label distribution
            unique, counts = np.unique(labels, return_counts=True)
            self.logger.info(f"Label distribution: SELL={counts[0]}, HOLD={counts[1]}, BUY={counts[2]}")
            
            # Prepare data
            X_seq, y_seq = self.features.prepare_sequences(features, labels)
            seq_len = self.config.SEQUENCE_LENGTH
            X_flat = self.features.scaler.transform(features.values)[seq_len:]
            y_flat = labels[seq_len:]
            
            if len(X_seq) < 3000:
                self.logger.warning(f"Not enough data: {len(X_seq)} (need 3000+)")
                return False
            
            self.logger.info(f"Training data: {len(X_seq)} samples")
            
            # Train TCN
            self.logger.info("-" * 30)
            self.logger.info("Training TCN...")
            self.tcn.build((X_seq.shape[1], X_seq.shape[2]))
            self.tcn.train(X_seq, y_seq, epochs=50)
            
            # Train LGBM
            self.logger.info("-" * 30)
            self.logger.info("Training LightGBM...")
            self.lgbm.train(X_flat, y_flat)
            
            self.trained = True
            self.logger.info("=" * 50)
            self.logger.info("TRAINING COMPLETE")
            self.logger.info("=" * 50)
            return True
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float, Dict]:
        """Get ensemble prediction"""
        if not self.trained:
            return 1, 0.0, {}
        
        with self.lock:
            features = self.features.calculate(df)
            X_seq, _ = self.features.prepare_sequences(features)
            X_flat = self.features.scaler.transform(features.values)
            
            if len(X_seq) == 0:
                return 1, 0.0, {}
            
            # Get predictions
            tcn_probs = self.tcn.predict(X_seq[-1:])
            lgbm_probs = self.lgbm.predict(X_flat[-1:])
            micro_sig, micro_conf = self.micro.predict(df)
            
            # Convert micro to probs
            micro_probs = np.array([0.33, 0.34, 0.33])
            if micro_sig != 1:
                micro_probs = np.array([0.1, 0.1, 0.1])
                micro_probs[micro_sig] = micro_conf
                micro_probs /= micro_probs.sum()
            
            # Weighted ensemble
            probs = (
                self.config.TCN_WEIGHT * tcn_probs[0] +
                self.config.LGBM_WEIGHT * lgbm_probs[0] +
                self.config.MICRO_WEIGHT * micro_probs
            )
            
            signal = int(np.argmax(probs))
            confidence = float(probs[signal])
            
            info = {
                'tcn': int(np.argmax(tcn_probs[0])),
                'lgbm': int(np.argmax(lgbm_probs[0])),
                'micro': micro_sig
            }
            
            return signal, confidence, info
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.tcn.save(f"{path}/model")
        self.lgbm.save(f"{path}/model")
        with open(f"{path}/scaler.pkl", 'wb') as f:
            pickle.dump(self.features.scaler, f)
    
    def load(self, path) -> bool:
        try:
            self.tcn.load(f"{path}/model")
            self.lgbm.load(f"{path}/model")
            with open(f"{path}/scaler.pkl", 'rb') as f:
                self.features.scaler = pickle.load(f)
                self.features.fitted = True
            self.trained = True
            return True
        except:
            return False



# =============================================================================
# MT5 EXECUTOR
# =============================================================================

class MT5:
    """MetaTrader 5 connection"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.symbol = config.SYMBOL
    
    def connect(self) -> bool:
        if not mt5.initialize():
            self.logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False
        
        # Find symbol
        for sym in self.config.SYMBOL_ALTERNATIVES:
            if mt5.symbol_info(sym):
                self.symbol = sym
                self.config.SYMBOL = sym
                break
        else:
            self.logger.error("No gold symbol found")
            return False
        
        mt5.symbol_select(self.symbol, True)
        info = mt5.account_info()
        self.logger.info(f"Connected | Account: {info.login} | Balance: ${info.balance:.2f} | Symbol: {self.symbol}")
        return True
    
    def disconnect(self):
        mt5.shutdown()
    
    def get_candles(self, count: int) -> Optional[pd.DataFrame]:
        rates = mt5.copy_rates_from_pos(self.symbol, self.config.TIMEFRAME, 0, count)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def get_price(self) -> Tuple[float, float]:
        tick = mt5.symbol_info_tick(self.symbol)
        return (tick.bid, tick.ask) if tick else (0.0, 0.0)
    
    def get_balance(self) -> float:
        info = mt5.account_info()
        return info.balance if info else 0.0
    
    def get_positions(self) -> List[Dict]:
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return []
        return [{'ticket': p.ticket, 'type': p.type, 'volume': p.volume, 
                 'price_open': p.price_open, 'profit': p.profit} for p in positions]
    
    def open_position(self, order_type: int, volume: float) -> Optional[int]:
        """Open position without SL/TP (VOM handles it)"""
        # Cooldown check to prevent rapid-fire order spam
        current_time = time.time()
        if hasattr(self, '_last_order_attempt'):
            if current_time - self._last_order_attempt < 1.0:  # 1 second min between attempts
                return None
        
        # Check if we're in error cooldown
        if hasattr(self, '_error_cooldown_until'):
            if current_time < self._error_cooldown_until:
                return None
        
        self._last_order_attempt = current_time
        
        info = mt5.symbol_info(self.symbol)
        volume = max(info.volume_min, min(volume, info.volume_max))
        volume = round(volume / info.volume_step) * info.volume_step
        volume = round(volume, 2)
        
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.ask if order_type == 0 else tick.bid
        mt5_type = mt5.ORDER_TYPE_BUY if order_type == 0 else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": mt5_type,
            "price": price,
            "deviation": 20,
            "magic": 123458,
            "comment": "HFT_V3",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            # Clear error cooldown on success
            if hasattr(self, '_error_cooldown_until'):
                del self._error_cooldown_until
            self.logger.info(f"OPENED: {result.order} | {'BUY' if order_type==0 else 'SELL'} | {volume} lots @ {price:.2f}")
            return result.order
        
        # Handle errors with appropriate cooldowns
        error_code = result.retcode if result else 0
        if error_code == 10031:
            self._error_cooldown_until = current_time + 10.0  # 10s cooldown for rejection
            self.logger.error(f"Open failed: {error_code} - cooling down 10s")
        elif error_code == 10004:
            self._error_cooldown_until = current_time + 2.0  # 2s for requote
            self.logger.warning(f"Requote - retrying in 2s")
        elif error_code == 10019:
            self._error_cooldown_until = current_time + 60.0  # 60s for no margin
            self.logger.error(f"Insufficient margin - cooling down 60s")
        else:
            self._error_cooldown_until = current_time + 5.0  # 5s default
            self.logger.error(f"Open failed: {error_code}")
        
        return None
    
    def close_position(self, ticket: int) -> Tuple[bool, float, float]:
        """Close position by ticket"""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False, 0.0, 0.0
        
        pos = positions[0]
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123458,
            "comment": "HFT_V3_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"CLOSED: {ticket} @ {price:.2f} | P&L: ${pos.profit:.2f}")
            return True, price, pos.profit
        
        return False, 0.0, 0.0


# =============================================================================
# MAIN BOT
# =============================================================================

class HFTBot:
    """Main trading bot"""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logging(self.config)
        self.mt5 = MT5(self.config, self.logger)
        self.ensemble = Ensemble(self.config, self.logger)
        self.vom = VOM(self.config, self.logger)
        self.running = False
        
        # Stats
        self.trades = 0
        self.wins = 0
        self.pnl = 0.0
        
        os.makedirs(self.config.MODELS_DIR, exist_ok=True)
    
    def calculate_lot(self, confidence: float, balance: float) -> float:
        """Simple lot sizing based on confidence and balance"""
        # Base: 0.01 per $100
        base = max(0.01, round(balance / 10000, 2))
        
        # Confidence multiplier
        if confidence >= 0.70:
            mult = 4.0
        elif confidence >= 0.60:
            mult = 3.0
        elif confidence >= 0.50:
            mult = 2.0
        elif confidence >= 0.40:
            mult = 1.5
        else:
            mult = 1.0
        
        lot = base * mult
        return max(self.config.MIN_LOT, min(lot, self.config.MAX_LOT))
    
    def calculate_tp_sl(self, entry: float, signal: int, atr: float) -> Tuple[float, float]:
        """Calculate TP/SL based on ATR"""
        pip = self.config.POINT_VALUE
        
        # Scale TP/SL with ATR
        atr_pips = atr / pip
        if atr_pips < 3:
            tp_pips, sl_pips = 80, 40
        elif atr_pips < 7:
            tp_pips, sl_pips = 150, 60
        elif atr_pips < 12:
            tp_pips, sl_pips = 250, 90
        else:
            tp_pips, sl_pips = 400, 120
        
        # Cap
        tp_pips = min(tp_pips, self.config.MAX_TP_PIPS)
        sl_pips = min(sl_pips, self.config.MAX_SL_PIPS)
        
        if signal == 2:  # BUY
            tp = entry + tp_pips * pip
            sl = entry - sl_pips * pip
        else:  # SELL
            tp = entry - tp_pips * pip
            sl = entry + sl_pips * pip
        
        return tp, sl
    
    def initialize(self) -> bool:
        self.logger.info("=" * 50)
        self.logger.info("XAUUSD HFT V3 - CLEAN VERSION")
        self.logger.info("=" * 50)
        
        if not self.mt5.connect():
            return False
        
        # Load or train models
        if self.ensemble.load(self.config.MODELS_DIR):
            self.logger.info("Loaded existing models")
        else:
            self.logger.info(f"Fetching {self.config.TRAINING_CANDLES} candles for training...")
            df = self.mt5.get_candles(self.config.TRAINING_CANDLES)
            if df is None or len(df) < 5000:
                self.logger.error(f"Not enough data: got {len(df) if df is not None else 0}, need 5000+")
                return False
            self.logger.info(f"Got {len(df)} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
            if not self.ensemble.train(df):
                return False
            self.ensemble.save(self.config.MODELS_DIR)
        
        self.logger.info("Ready to trade")
        return True
    
    def run(self):
        self.running = True
        self.logger.info("Starting trading loop...")
        last_status = datetime.now()
        
        try:
            while self.running:
                self._cycle()
                
                # Status every 60s
                if (datetime.now() - last_status).seconds >= 60:
                    self._log_status()
                    last_status = datetime.now()
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("Stopping...")
        finally:
            self._shutdown()
    
    def _cycle(self):
        """Main trading cycle"""
        try:
            bid, ask = self.mt5.get_price()
            if bid == 0:
                return
            
            # Manage existing positions
            self._manage_positions(bid, ask)
            
            # Check position limit
            if self.vom.count() >= self.config.MAX_POSITIONS:
                return
            
            # Get signal
            df = self.mt5.get_candles(200)
            if df is None:
                return
            
            signal, confidence, info = self.ensemble.predict(df)
            
            # Skip HOLD or low confidence
            if signal == 1 or confidence < self.config.MIN_CONFIDENCE:
                return
            
            # Calculate lot and TP/SL
            balance = self.mt5.get_balance()
            lot = self.calculate_lot(confidence, balance)
            
            features = self.ensemble.features.calculate(df)
            atr = features['atr'].iloc[-1]
            
            entry = ask if signal == 2 else bid
            tp, sl = self.calculate_tp_sl(entry, signal, atr)
            
            # Open trade
            direction = "BUY" if signal == 2 else "SELL"
            self.logger.info(f"SIGNAL: {direction} | Conf: {confidence:.1%} | Lot: {lot} | TCN:{info['tcn']} LGBM:{info['lgbm']} Micro:{info['micro']}")
            
            ticket = self.mt5.open_position(signal - 1 if signal == 2 else 1, lot)  # 0=BUY, 1=SELL
            if ticket:
                pos = Position(
                    ticket=ticket,
                    order_type=0 if signal == 2 else 1,
                    volume=lot,
                    entry_price=entry,
                    entry_time=datetime.now(),
                    sl_price=sl,
                    tp_price=tp,
                    confidence=confidence
                )
                self.vom.add(pos)
                
        except Exception as e:
            self.logger.error(f"Cycle error: {e}")
    
    def _manage_positions(self, bid: float, ask: float):
        """Check and close positions"""
        mt5_positions = {p['ticket'] for p in self.mt5.get_positions()}
        
        for pos in self.vom.get_all():
            # Sync check
            if pos.ticket not in mt5_positions:
                self.logger.warning(f"Position {pos.ticket} closed externally")
                self.vom.remove(pos.ticket)
                continue
            
            # Check price
            price = bid if pos.order_type == 0 else ask
            should_close, reason = self.vom.check_exit(pos.ticket, price)
            
            if should_close:
                success, close_price, profit = self.mt5.close_position(pos.ticket)
                if success:
                    self.vom.remove(pos.ticket)
                    self.trades += 1
                    self.pnl += profit
                    if profit > 0:
                        self.wins += 1
                    
                    pip = self.config.POINT_VALUE
                    pips = (close_price - pos.entry_price) / pip if pos.order_type == 0 else (pos.entry_price - close_price) / pip
                    self.logger.info(f"EXIT: {pos.ticket} | {reason} | {pips:+.1f} pips | ${profit:.2f}")
    
    def _log_status(self):
        """Log current status"""
        positions = self.vom.get_all()
        bid, _ = self.mt5.get_price()
        
        win_rate = (self.wins / self.trades * 100) if self.trades > 0 else 0
        self.logger.info(f"STATUS: {len(positions)} positions | Trades: {self.trades} | Win: {win_rate:.1f}% | P&L: ${self.pnl:.2f}")
        
        for pos in positions:
            pip = self.config.POINT_VALUE
            pips = (bid - pos.entry_price) / pip if pos.order_type == 0 else (pos.entry_price - bid) / pip
            trail = f"Trail: {pos.trailing_sl:.2f}" if pos.trailing_active else ""
            self.logger.info(f"  {pos.ticket} | {'BUY' if pos.order_type==0 else 'SELL'} | {pips:+.1f} pips | {trail}")
    
    def _shutdown(self):
        """Clean shutdown"""
        self.running = False
        
        # Close all positions
        for pos in self.vom.get_all():
            self.mt5.close_position(pos.ticket)
            self.vom.remove(pos.ticket)
        
        self.logger.info(f"FINAL: Trades: {self.trades} | Wins: {self.wins} | P&L: ${self.pnl:.2f}")
        self.mt5.disconnect()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("=" * 50)
    print("XAUUSD HFT V3 - CLEAN VERSION")
    print("=" * 50)
    
    bot = HFTBot()
    if bot.initialize():
        bot.run()
    else:
        print("Initialization failed")


if __name__ == "__main__":
    main()
