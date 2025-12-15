"""
================================================================================
STRATOSPHERE - XAUUSD HFT SCALPING BOT v2.0
================================================================================
CRITICAL FIXES IMPLEMENTED:
1. VOMS (Virtual Order Management System) - Bypasses 2000 msg/day limit
2. Threaded ML Engine - Solves Python GIL latency issue
3. Volatility Gating - Prevents commission drag death spiral
4. Unlimited Leverage Mode - Aggressive position sizing for Exness

Target: 1500-2500 trades/day | 70%+ Winrate | Exness Raw Spread
================================================================================
"""

import os
import sys
import time
import threading
import queue
import argparse
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib

# MT5 Import
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    print("⚠️ MetaTrader5 not installed")

# ML Imports
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    HAS_TF = True
except ImportError:
    HAS_TF = False

import ta
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange


# ==================== CONFIGURATION ====================
@dataclass
class StratosphereConfig:
    """Stratosphere Bot Configuration - Hardcoded Aggressive Mode"""
    
    # Symbol Settings - Multi-symbol support
    SYMBOLS: tuple = ("XAUUSDm", "BTCUSDm")  # Exness micro account symbols
    TIMEFRAME: int = 1  # M1
    MAGIC_NUMBER: int = 777777
    
    # === RISK MANAGEMENT (UNLIMITED LEVERAGE MODE) ===
    LEVERAGE_MODE: str = "UNLIMITED"  # Exness 1:Unlimited
    RISK_PER_TRADE_PCT: float = 0.02  # 2% risk per trade (aggressive)
    MAX_POSITION_PCT: float = 0.50  # Max 50% of balance in margin
    MAX_CONCURRENT_POSITIONS: int = 2  # One per symbol
    
    # === VIRTUAL ORDER MANAGEMENT (VOMS) ===
    # NO SL/TP sent to broker - managed in memory
    VIRTUAL_SL_PIPS: float = 10.0  # 10 pips = $1.00 per 0.01 lot
    VIRTUAL_TP_PIPS: float = 20.0  # 20 pips = $2.00 per 0.01 lot (1:2 RR)
    VIRTUAL_TRAIL_TRIGGER_PIPS: float = 8.0  # Trail after 8 pips profit
    VIRTUAL_TRAIL_DISTANCE_PIPS: float = 5.0  # Trail 5 pips behind
    QUICK_EXIT_SECONDS: int = 180  # Close losers after 3 min
    
    # === COMMISSION & VOLATILITY GATING ===
    COMMISSION_PER_LOT: float = 7.0  # $7 per lot round trip (Exness Raw)
    MIN_ATR_PIPS: float = 8.0  # Minimum ATR to cover commission + profit
    PIP_VALUE: float = 0.01  # 1 pip = 0.01 for XAUUSD
    USD_PER_PIP_PER_LOT: float = 10.0  # $10 per pip per 1.0 lot
    
    # === ML SETTINGS ===
    LSTM_TIMESTEPS: int = 30  # Extended to 30 candles (30 min horizon on M1)
    ML_THRESHOLD: float = 0.65  # Slightly lower threshold for more signals
    RETRAIN_INTERVAL: int = 500  # Retrain more frequently
    TRAIN_LOOKBACK: int = 7000  # More historical data
    RETRAIN_CYCLES: int = 5  # Multiple training cycles for better accuracy
    PREDICTION_HORIZON: int = 30  # Predict 30 candles ahead (30 min)
    
    # === BTC-SPECIFIC ML SETTINGS (Optimized TCN+LightGBM v3.0) ===
    # Dynamic threshold range: 0.53-0.62 (auto-adjusted by volatility/session)
    BTC_PREDICTION_HORIZON: int = 2      # 1-3 candles ahead
    BTC_CONFIDENCE_THRESHOLD: float = 0.55  # Base threshold (dynamic: 0.53-0.62)
    BTC_LSTM_TIMESTEPS: int = 30         # TCN sequence length (30-60 optimal)
    BTC_RETRAIN_CYCLES: int = 3          # Randomized init each cycle
    BTC_MIN_ATR_USD: float = 20.0        # Min volatility in USD
    
    # === SESSION TIMES (GMT) ===
    LONDON_START: int = 8
    LONDON_END: int = 12
    NY_START: int = 13
    NY_END: int = 17
    
    # === THREADING ===
    INFERENCE_QUEUE_SIZE: int = 100
    TICK_SLEEP_MS: float = 1  # 1ms tick loop (0.001s)
    
    # === PATHS ===
    MODEL_DIR: str = "stratosphere_models"
    LOG_FILE: str = "stratosphere_trades.csv"


CFG = StratosphereConfig()


# ==================== LOGGING ====================
# Fix Windows console encoding for emoji characters
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler('stratosphere.log', encoding='utf-8')
    ]
)
log = logging.getLogger("STRATOSPHERE")


# ==================== SHARED MEMORY (Thread-Safe Signal Passing) ====================
class SignalType(Enum):
    NONE = 0
    LONG = 1
    SHORT = 2


@dataclass
class SharedSignal:
    """Thread-safe signal container between Brain and Reflex"""
    signal: SignalType = SignalType.NONE
    confidence: float = 0.0
    lstm_prob: float = 0.0
    xgb_prob: float = 0.0
    atr: float = 0.0
    timestamp: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update(self, signal: SignalType, confidence: float, lstm_prob: float, 
               xgb_prob: float, atr: float):
        with self.lock:
            self.signal = signal
            self.confidence = confidence
            self.lstm_prob = lstm_prob
            self.xgb_prob = xgb_prob
            self.atr = atr
            self.timestamp = time.time()
    
    def read(self) -> Tuple[SignalType, float, float]:
        with self.lock:
            return self.signal, self.confidence, self.atr
    
    def clear(self):
        with self.lock:
            self.signal = SignalType.NONE


# ==================== VIRTUAL ORDER MANAGEMENT SYSTEM (VOMS) ====================
@dataclass
class VirtualPosition:
    """Virtual position tracked in memory - NOT sent to broker"""
    ticket: int
    symbol: str  # Symbol for multi-symbol support
    order_type: str  # 'BUY' or 'SELL'
    entry_price: float
    lot_size: float
    virtual_sl: float
    virtual_tp: float
    entry_time: datetime
    trailing_active: bool = False
    trailing_sl: float = 0.0


class VOMS:
    """
    Virtual Order Management System
    
    CRITICAL: This system does NOT send SL/TP to broker.
    All risk management is handled in Python memory.
    Only ORDER_OPEN and ORDER_CLOSE are sent to broker.
    This reduces API messages by >95% to avoid Exness bans.
    """
    
    def __init__(self):
        self.positions: Dict[int, VirtualPosition] = {}
        self.lock = threading.Lock()
        self.api_calls_today = 0
        self.last_reset = datetime.utcnow().date()
    
    def add_position(self, ticket: int, symbol: str, order_type: str, entry_price: float, 
                     lot_size: float) -> VirtualPosition:
        """Register new position with virtual SL/TP"""
        
        pip = CFG.PIP_VALUE
        
        if order_type == 'BUY':
            virtual_sl = entry_price - (CFG.VIRTUAL_SL_PIPS * pip)
            virtual_tp = entry_price + (CFG.VIRTUAL_TP_PIPS * pip)
        else:
            virtual_sl = entry_price + (CFG.VIRTUAL_SL_PIPS * pip)
            virtual_tp = entry_price - (CFG.VIRTUAL_TP_PIPS * pip)
        
        pos = VirtualPosition(
            ticket=ticket,
            symbol=symbol,
            order_type=order_type,
            entry_price=entry_price,
            lot_size=lot_size,
            virtual_sl=round(virtual_sl, 2),
            virtual_tp=round(virtual_tp, 2),
            entry_time=datetime.utcnow()
        )
        
        with self.lock:
            self.positions[ticket] = pos
        
        log.info(f"VOMS | {symbol} | {order_type} {lot_size} @ {entry_price} | "
                f"vSL: {pos.virtual_sl} | vTP: {pos.virtual_tp}")
        
        return pos
    
    def check_exits(self, get_tick_func) -> List[Tuple[int, str, str]]:
        """
        Check all positions for virtual SL/TP/Trail hits.
        Returns list of (ticket, symbol, reason) to close.
        get_tick_func: function(symbol) -> (bid, ask, spread)
        """
        exits = []
        
        with self.lock:
            for ticket, pos in list(self.positions.items()):
                bid, ask, _ = get_tick_func(pos.symbol)
                if bid == 0:
                    continue
                    
                current_price = bid if pos.order_type == 'BUY' else ask
                
                # Calculate profit in pips
                if pos.order_type == 'BUY':
                    profit_pips = (current_price - pos.entry_price) / CFG.PIP_VALUE
                else:
                    profit_pips = (pos.entry_price - current_price) / CFG.PIP_VALUE
                
                # Check virtual TP
                if pos.order_type == 'BUY' and current_price >= pos.virtual_tp:
                    exits.append((ticket, pos.symbol, 'TP'))
                    continue
                elif pos.order_type == 'SELL' and current_price <= pos.virtual_tp:
                    exits.append((ticket, pos.symbol, 'TP'))
                    continue
                
                # Check virtual SL (or trailing SL)
                active_sl = pos.trailing_sl if pos.trailing_active else pos.virtual_sl
                
                if pos.order_type == 'BUY' and current_price <= active_sl:
                    exits.append((ticket, pos.symbol, 'SL' if not pos.trailing_active else 'TRAIL'))
                    continue
                elif pos.order_type == 'SELL' and current_price >= active_sl:
                    exits.append((ticket, pos.symbol, 'SL' if not pos.trailing_active else 'TRAIL'))
                    continue
                
                # Check quick exit (losers held too long)
                time_held = (datetime.utcnow() - pos.entry_time).total_seconds()
                if time_held > CFG.QUICK_EXIT_SECONDS and profit_pips < 0:
                    exits.append((ticket, pos.symbol, 'QUICK_EXIT'))
                    continue
                
                # Update trailing stop
                if profit_pips >= CFG.VIRTUAL_TRAIL_TRIGGER_PIPS:
                    self._update_trailing(pos, current_price)
        
        return exits
    
    def _update_trailing(self, pos: VirtualPosition, current_price: float):
        """Update trailing stop level"""
        pip = CFG.PIP_VALUE
        
        if pos.order_type == 'BUY':
            new_trail = current_price - (CFG.VIRTUAL_TRAIL_DISTANCE_PIPS * pip)
            if not pos.trailing_active or new_trail > pos.trailing_sl:
                pos.trailing_sl = round(new_trail, 2)
                pos.trailing_active = True
        else:
            new_trail = current_price + (CFG.VIRTUAL_TRAIL_DISTANCE_PIPS * pip)
            if not pos.trailing_active or new_trail < pos.trailing_sl:
                pos.trailing_sl = round(new_trail, 2)
                pos.trailing_active = True
    
    def remove_position(self, ticket: int):
        """Remove closed position"""
        with self.lock:
            if ticket in self.positions:
                del self.positions[ticket]
    
    def get_position_count(self, symbol: str = None) -> int:
        with self.lock:
            if symbol:
                return sum(1 for p in self.positions.values() if p.symbol == symbol)
            return len(self.positions)
    
    def increment_api_calls(self):
        """Track API calls for rate limiting awareness"""
        today = datetime.utcnow().date()
        if today != self.last_reset:
            self.api_calls_today = 0
            self.last_reset = today
        self.api_calls_today += 1
        
        if self.api_calls_today > 1800:  # Warning at 90% of limit
            log.warning(f"⚠️ API calls today: {self.api_calls_today}/2000")


# ==================== MT5 HANDLER (Minimal API Calls) ====================
class MT5Handler:
    """MT5 connection with minimal API footprint - Multi-symbol support"""
    
    def __init__(self):
        self.connected = False
        self.symbol_info: Dict[str, any] = {}  # Per-symbol info
        self.active_symbols: List[str] = []
        
    def connect(self) -> bool:
        if not HAS_MT5:
            log.error("MetaTrader5 not installed")
            return False
        
        for attempt in range(3):
            if mt5.initialize():
                self.connected = True
                info = mt5.account_info()
                
                log.info(f"MT5 Connected | Account: {info.login} | "
                        f"Balance: ${info.balance:.2f} | Leverage: 1:{info.leverage}")
                
                # Enable all symbols in Market Watch (use runtime ACTIVE_SYMBOLS)
                symbols_to_use = get_active_symbols()
                log.info(f"Symbols to initialize: {symbols_to_use}")
                for symbol in symbols_to_use:
                    if mt5.symbol_select(symbol, True):
                        sym_info = mt5.symbol_info(symbol)
                        if sym_info:
                            self.symbol_info[symbol] = sym_info
                            self.active_symbols.append(symbol)
                            log.info(f"✅ {symbol} | Spread: {sym_info.spread} | Min Lot: {sym_info.volume_min}")
                        else:
                            log.warning(f"⚠️ {symbol} info not available")
                    else:
                        log.warning(f"⚠️ Could not select {symbol} in Market Watch")
                
                if not self.active_symbols:
                    log.error("No symbols available! Check broker symbol names.")
                    return False
                
                return True
            time.sleep(1)
        
        log.error("MT5 connection failed")
        return False
    
    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
    
    def get_tick(self, symbol: str) -> Tuple[float, float, float]:
        """Get current bid, ask, spread for symbol"""
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return tick.bid, tick.ask, tick.ask - tick.bid
        return 0, 0, 0
    
    def get_balance(self) -> float:
        info = mt5.account_info()
        return info.balance if info else 0
    
    def get_equity(self) -> float:
        info = mt5.account_info()
        return info.equity if info else 0
    
    def fetch_candles(self, symbol: str, count: int = 200) -> pd.DataFrame:
        """Fetch M1 OHLCV for symbol"""
        mt5.symbol_select(symbol, True)
        
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            log.warning(f"{symbol} copy_rates_from_pos failed: {error}")
            from datetime import datetime, timezone
            utc_now = datetime.now(timezone.utc)
            rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, utc_now, count)
            if rates is None or len(rates) == 0:
                log.error(f"{symbol} data fetch failed: {mt5.last_error()}")
                return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def open_position(self, symbol: str, order_type: str, lot: float) -> Optional[int]:
        """
        Open position with NO SL/TP (VOMS handles risk).
        Returns ticket or None.
        """
        bid, ask, _ = self.get_tick(symbol)
        
        if order_type == "BUY":
            price = ask
            mt5_type = mt5.ORDER_TYPE_BUY
        else:
            price = bid
            mt5_type = mt5.ORDER_TYPE_SELL
        
        # Validate lot size
        sym_info = self.symbol_info.get(symbol)
        if sym_info:
            lot = max(sym_info.volume_min, min(round(lot, 2), sym_info.volume_max))
        else:
            lot = max(0.01, min(round(lot, 2), 10.0))
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5_type,
            "price": price,
            "sl": 0.0,  # NO SL - VOMS handles this
            "tp": 0.0,  # NO TP - VOMS handles this
            "deviation": 20,
            "magic": CFG.MAGIC_NUMBER,
            "comment": "STRATOSPHERE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"🟢 OPEN {symbol} | {order_type} {lot} lots @ {price:.2f} | Ticket: {result.order}")
            return result.order
        
        error = mt5.last_error() if result is None else result.retcode
        log.error(f"{symbol} order failed: {error}")
        return None
    
    def close_position(self, ticket: int, symbol: str, reason: str = "") -> bool:
        """Close position by ticket"""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        
        pos = positions[0]
        bid, ask, _ = self.get_tick(symbol)
        
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        close_price = bid if pos.type == 0 else ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "deviation": 20,
            "magic": CFG.MAGIC_NUMBER,
            "comment": f"CLOSE_{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"🔴 CLOSE {symbol} | Ticket {ticket} | Reason: {reason} | P&L: ${pos.profit:.2f}")
            return True
        
        return False
    
    def get_open_positions(self, symbol: str = None) -> List[Dict]:
        """Get positions for our magic number, optionally filtered by symbol"""
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            # Get all positions for our symbols
            all_positions = []
            for sym in self.active_symbols:
                pos = mt5.positions_get(symbol=sym)
                if pos:
                    all_positions.extend(pos)
            positions = all_positions if all_positions else None
        
        if not positions:
            return []
        return [{'ticket': p.ticket, 'symbol': p.symbol, 'type': 'BUY' if p.type == 0 else 'SELL',
                 'volume': p.volume, 'price': p.price_open, 'profit': p.profit}
                for p in positions if p.magic == CFG.MAGIC_NUMBER]


# ==================== INDICATORS ENGINE ====================
class IndicatorEngine:
    """Technical indicator calculations"""
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
        
        # EMAs
        df['ema9'] = EMAIndicator(close, 9).ema_indicator()
        df['ema21'] = EMAIndicator(close, 21).ema_indicator()
        df['ema50'] = EMAIndicator(close, 50).ema_indicator()
        
        # RSI
        df['rsi14'] = RSIIndicator(close, 14).rsi()
        
        # MACD
        macd = MACD(close)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close, 20)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / bb.bollinger_mavg()
        
        # ATR (Critical for volatility gating)
        df['atr10'] = AverageTrueRange(high, low, close, 10).average_true_range()
        df['atr14'] = AverageTrueRange(high, low, close, 14).average_true_range()
        df['atr_pips'] = df['atr10'] / CFG.PIP_VALUE  # Convert to pips
        
        # ADX
        adx = ADXIndicator(high, low, close, 14)
        df['adx'] = adx.adx()
        df['di_plus'] = adx.adx_pos()
        df['di_minus'] = adx.adx_neg()
        
        # Stochastic
        stoch = StochasticOscillator(high, low, close, 14)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volume
        df['vol_sma20'] = volume.rolling(20).mean()
        df['vol_ratio'] = volume / df['vol_sma20']
        
        # Trend
        df['trend'] = np.where(close > df['ema50'], 1, -1)
        
        # Momentum
        df['momentum'] = close.diff(10)
        df['roc'] = close.pct_change(10) * 100
        
        return df.dropna()


# ==================== BTC MICROSTRUCTURE ML ENGINE ====================
# Import BTC-specific predictor for microstructure-based HFT
try:
    from models.btc_hft_predictor import BTCHFTPredictor, BTCMicrostructureFeatures
    HAS_BTC_HFT = True
except ImportError:
    HAS_BTC_HFT = False
    log.warning("BTCHFTPredictor not available - using standard ML for BTC")


class BTCMicrostructureMLEngine:
    """
    BTC-specific ML engine using OPTIMIZED TCN + LightGBM ensemble v3.0.
    
    OPTIMIZATION LAYERS:
    1. Dynamic Confidence Thresholds (0.53-0.62 based on volatility)
    2. Volatility Regime Filtering (skip low-vol, adjust for extreme)
    3. Session-Based Trade Gating (UTC windows)
    4. Optimized TCN (dilation, dropout 0.18, L2 reg)
    5. Optimized LightGBM (feature bagging, min_child_weight)
    6. Orthogonal Microstructure Features Only (20 features)
    7. Flat Prediction Filtering (margin_min = 0.07)
    8. Adaptive TP/SL based on ATR
    
    Target: Higher win rate, cleaner trade flow, better expectancy
    """
    
    def __init__(self, shared_signal: 'SharedSignal'):
        self.shared_signal = shared_signal
        self.predictor = None
        self.is_trained = False
        self.running = False
        self.data_queue = queue.Queue(maxsize=CFG.INFERENCE_QUEUE_SIZE)
        self.thread = None
        self.candles_processed = 0
        self.symbol = "BTCUSDm"
        
        os.makedirs(CFG.MODEL_DIR, exist_ok=True)
    
    def start(self):
        """Start background inference thread"""
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        log.info("🧠 BTC Optimized TCN+LightGBM v3.0 started")
    
    def stop(self):
        """Stop inference thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def submit_data(self, df: pd.DataFrame):
        """Submit data for inference (non-blocking)"""
        try:
            self.data_queue.put_nowait(df)
        except queue.Full:
            pass
    
    def _inference_loop(self):
        """Background inference loop"""
        while self.running:
            try:
                df = self.data_queue.get(timeout=0.1)
                self._process_inference(df)
                self.candles_processed += 1
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"BTC inference error: {e}")
    
    def _process_inference(self, df: pd.DataFrame):
        """
        Run optimized inference with all filtering layers.
        The predictor handles: volatility regime, session, threshold, flat filter
        """
        if not self.is_trained or self.predictor is None:
            return
        
        try:
            # Get prediction (includes all optimization layers)
            result = self.predictor.predict(df)
            
            signal = SignalType.NONE
            confidence = result.get('confidence', 0.5)
            lgbm_prob = result.get('lgbm_prob', 0.5)
            tcn_prob = result.get('tcn_prob', 0.5) or 0.5
            
            # Extract ATR from result or calculate
            atr = 0
            if 'tp_sl' in result and result['tp_sl']:
                atr = result['tp_sl'].get('sl_pips', 0) / 1.3  # Reverse calculate
            
            # Only trade if gated (passed all filters)
            if result.get('gated', False):
                if result['signal'] == 'BUY':
                    signal = SignalType.LONG
                elif result['signal'] == 'SELL':
                    signal = SignalType.SHORT
            
            # Enhanced debug logging every 60 candles
            if self.candles_processed % 60 == 0:
                regime = result.get('vol_regime', 'unknown')
                threshold = result.get('threshold', 0.55)
                reason = result.get('reason', '')
                stats = result.get('filter_stats', {})
                
                log.info(f"BTC | Regime: {regime} | Thresh: {threshold:.2f} | "
                        f"LGBM: {lgbm_prob:.1%} | TCN: {tcn_prob:.1%} | "
                        f"Signal: {result['signal']} | "
                        f"Filtered: {stats.get('filtered', 0)}/{stats.get('passed', 0) + stats.get('filtered', 0)}")
                
                if reason:
                    log.info(f"     Skip reason: {reason}")
            
            self.shared_signal.update(signal, confidence, tcn_prob, lgbm_prob, atr)
            
        except Exception as e:
            log.error(f"BTC prediction error: {e}")
            self.shared_signal.update(SignalType.NONE, 0, 0, 0, 0)
    
    def train(self, df: pd.DataFrame, symbol: str = "BTCUSDm"):
        """Train optimized BTC predictor"""
        self.symbol = symbol
        log.info(f"🔄 Training BTC Optimized TCN+LightGBM v3.0 for {symbol}...")
        log.info(f"   Horizon: {CFG.BTC_PREDICTION_HORIZON} candles | Base Threshold: {CFG.BTC_CONFIDENCE_THRESHOLD}")
        log.info(f"   Optimizations: Dynamic threshold, Vol regime, Session filter, Flat filter")
        
        try:
            self.predictor = BTCHFTPredictor(
                prediction_horizon=CFG.BTC_PREDICTION_HORIZON,
                base_threshold=CFG.BTC_CONFIDENCE_THRESHOLD,
                use_tcn=HAS_TF,
                tcn_timesteps=CFG.BTC_LSTM_TIMESTEPS,
                symbol=symbol
            )
            
            metrics = self.predictor.train(df, retrain_cycles=CFG.BTC_RETRAIN_CYCLES)
            
            log.info(f"✅ BTC LightGBM Accuracy: {metrics['lgbm_accuracy']:.2%}")
            if metrics.get('tcn_accuracy'):
                log.info(f"✅ BTC TCN Accuracy: {metrics['tcn_accuracy']:.2%}")
            log.info(f"   Features: {metrics['features']} (orthogonal only)")
            log.info(f"   Samples: {metrics['samples_train']} train / {metrics['samples_val']} val")
            
            # Save model
            model_path = os.path.join(CFG.MODEL_DIR, f"btc_hft_{symbol}.pkl")
            self.predictor.save(model_path)
            log.info(f"💾 BTC model saved: {model_path}")
            
            self.is_trained = True
            self.candles_processed = 0
            
        except Exception as e:
            log.error(f"BTC training failed: {e}")
            import traceback
            traceback.print_exc()
            self.is_trained = False
    
    def should_retrain(self) -> bool:
        return self.candles_processed >= CFG.RETRAIN_INTERVAL
    
    def get_filter_stats(self) -> dict:
        """Get trade filtering statistics from predictor"""
        if self.predictor:
            return self.predictor.get_filter_stats()
        return {'total_signals': 0, 'passed': 0, 'filtered': 0, 'filter_rate': 0}


# ==================== THREADED ML ENGINE (The "Brain") ====================
class ThreadedMLEngine:
    """
    Asynchronous ML inference engine.
    Runs in background thread to avoid blocking tick processing.
    """
    
    def __init__(self, shared_signal: SharedSignal):
        self.shared_signal = shared_signal
        self.lstm_model = None
        self.xgb_model = None
        self.lstm_scaler = MinMaxScaler()
        self.xgb_scaler = MinMaxScaler()
        self.is_trained = False
        self.running = False
        self.data_queue = queue.Queue(maxsize=CFG.INFERENCE_QUEUE_SIZE)
        self.thread = None
        self.candles_processed = 0
        
        os.makedirs(CFG.MODEL_DIR, exist_ok=True)
    
    def start(self):
        """Start background inference thread"""
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        log.info("🧠 ML Engine started (background thread)")
    
    def stop(self):
        """Stop inference thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def submit_data(self, df: pd.DataFrame):
        """Submit data for inference (non-blocking)"""
        try:
            self.data_queue.put_nowait(df)
        except queue.Full:
            pass  # Drop if queue full
    
    def _inference_loop(self):
        """Background inference loop"""
        while self.running:
            try:
                df = self.data_queue.get(timeout=0.1)
                self._process_inference(df)
                self.candles_processed += 1
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Inference error: {e}")
    
    def _process_inference(self, df: pd.DataFrame):
        """Run LSTM + XGBoost inference"""
        if not self.is_trained:
            return
        
        # Get ATR for volatility gating
        atr_pips = df['atr_pips'].iloc[-1] if 'atr_pips' in df.columns else 0
        
        # VOLATILITY GATE: Reject if ATR too low
        if atr_pips < CFG.MIN_ATR_PIPS:
            self.shared_signal.update(SignalType.NONE, 0, 0, 0, atr_pips)
            return
        
        # LSTM prediction
        lstm_short, lstm_long = self._lstm_predict(df)
        
        # XGBoost prediction
        xgb_short, xgb_long = self._xgb_predict(df)
        
        # Ensemble decision
        signal = SignalType.NONE
        confidence = 0.0
        
        # Both must agree with >70% probability
        if lstm_long > CFG.ML_THRESHOLD and xgb_long > CFG.ML_THRESHOLD:
            # Additional trend filter
            if df['close'].iloc[-1] > df['ema50'].iloc[-1]:
                signal = SignalType.LONG
                confidence = (lstm_long + xgb_long) / 2
        
        elif lstm_short > CFG.ML_THRESHOLD and xgb_short > CFG.ML_THRESHOLD:
            if df['close'].iloc[-1] < df['ema50'].iloc[-1]:
                signal = SignalType.SHORT
                confidence = (lstm_short + xgb_short) / 2
        
        self.shared_signal.update(signal, confidence, lstm_long, xgb_long, atr_pips)
    
    def _lstm_predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        """LSTM inference"""
        if self.lstm_model is None:
            return 0.5, 0.5
        
        try:
            features = df[['close', 'high', 'low', 'volume']].tail(CFG.LSTM_TIMESTEPS).values
            scaled = self.lstm_scaler.transform(features)
            X = scaled.reshape(1, CFG.LSTM_TIMESTEPS, 4)
            pred = self.lstm_model.predict(X, verbose=0)[0]
            return float(pred[0]), float(pred[1])
        except:
            return 0.5, 0.5
    
    def _xgb_predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        """XGBoost inference"""
        if self.xgb_model is None:
            return 0.5, 0.5
        
        try:
            features = getattr(self, 'xgb_features', ['ema9', 'ema21', 'rsi14', 'atr10', 'vol_sma20', 'macd', 'bb_width'])
            X = df[features].iloc[-1:].values
            scaled = self.xgb_scaler.transform(X)
            proba = self.xgb_model.predict_proba(scaled)[0]
            return float(proba[0]), float(proba[1])
        except Exception as e:
            return 0.5, 0.5

    def _build_lstm_model(self):
        """Build LSTM architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(CFG.LSTM_TIMESTEPS, 4)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, df: pd.DataFrame, symbol: str = ""):
        """Train both models with warm start - loads existing weights if available"""
        is_retrain = self.is_trained  # Check if this is a retrain (not first time)
        log.info(f"🔄 {'Retraining' if is_retrain else 'Training'} ML models for {symbol}... ({CFG.RETRAIN_CYCLES} cycles with warm start)")
        
        horizon = CFG.PREDICTION_HORIZON  # 30 candle prediction horizon
        
        # Prepare LSTM data with extended horizon
        lstm_features = df[['close', 'high', 'low', 'volume']].values
        self.lstm_scaler.fit(lstm_features)
        scaled = self.lstm_scaler.transform(lstm_features)
        
        X_lstm, y_lstm = [], []
        for i in range(CFG.LSTM_TIMESTEPS, len(scaled) - horizon):
            X_lstm.append(scaled[i - CFG.LSTM_TIMESTEPS:i])
            # Target: price direction over next 30 candles
            future_price = df['close'].iloc[i + horizon]
            current_price = df['close'].iloc[i]
            target = 1 if future_price > current_price else 0
            y_lstm.append([1 - target, target])
        
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        
        best_lstm_acc = 0
        best_weights = None
        
        if len(X_lstm) > 100 and HAS_TF:
            X_train, X_val, y_train, y_val = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)
            
            # Build initial model
            model = self._build_lstm_model()
            
            # === WARM START: Load existing weights if available ===
            lstm_weights_path = os.path.join(CFG.MODEL_DIR, f"lstm_{symbol}.weights.h5")
            if os.path.exists(lstm_weights_path):
                try:
                    model.load_weights(lstm_weights_path)
                    # Evaluate loaded model to get baseline accuracy
                    baseline_acc = model.evaluate(X_val, y_val, verbose=0)[1]
                    best_lstm_acc = baseline_acc
                    best_weights = model.get_weights()
                    log.info(f"  🔥 Loaded existing LSTM weights | Baseline acc: {baseline_acc:.2%}")
                except Exception as e:
                    log.warning(f"  Could not load LSTM weights: {e}")
            
            # Multiple training cycles with WARM START
            for cycle in range(CFG.RETRAIN_CYCLES):
                # Warm start: load best weights from previous cycle
                if best_weights is not None:
                    model.set_weights(best_weights)
                    if cycle > 0:
                        log.info(f"  🔥 Warm start from best weights (acc: {best_lstm_acc:.2%})")
                
                # Train with early stopping
                early_stop = keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', patience=5, restore_best_weights=True
                )
                
                # Vary learning rate and epochs per cycle
                if cycle > 0 or best_weights is not None:
                    # Fine-tune with lower learning rate after first cycle or when warm starting
                    lr = 0.001 / (cycle + 2) if best_weights is not None else 0.001 / (cycle + 1)
                    model.optimizer.learning_rate.assign(lr)
                
                model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=50 + cycle * 10, batch_size=64, verbose=0, callbacks=[early_stop])
                
                acc = model.evaluate(X_val, y_val, verbose=0)[1]
                log.info(f"  LSTM Cycle {cycle+1}/{CFG.RETRAIN_CYCLES} | Accuracy: {acc:.2%}")
                
                if acc > best_lstm_acc:
                    best_lstm_acc = acc
                    best_weights = model.get_weights()
                    self.lstm_model = keras.models.clone_model(model)
                    self.lstm_model.set_weights(best_weights)
                    self.lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            log.info(f"✅ LSTM Best Accuracy: {best_lstm_acc:.2%}")
            
            # Save best model weights
            model_path = os.path.join(CFG.MODEL_DIR, f"lstm_{symbol}.weights.h5")
            try:
                self.lstm_model.save_weights(model_path)
                log.info(f"💾 LSTM weights saved: {model_path}")
            except Exception as e:
                log.warning(f"Could not save LSTM weights: {e}")
        
        # Prepare XGBoost data with extended horizon
        xgb_features = ['ema9', 'ema21', 'rsi14', 'atr10', 'vol_sma20', 'macd', 'bb_width', 'adx', 'momentum', 'roc']
        available_features = [f for f in xgb_features if f in df.columns]
        X_xgb = df[available_features].values[:-horizon]
        
        # Target: price direction over next 30 candles
        y_xgb = (df['close'].shift(-horizon) > df['close']).astype(int).values[:-horizon]
        
        self.xgb_scaler.fit(X_xgb)
        X_xgb_scaled = self.xgb_scaler.transform(X_xgb)
        
        X_train, X_val, y_train, y_val = train_test_split(X_xgb_scaled, y_xgb, test_size=0.2, shuffle=False)
        
        best_xgb_acc = 0
        best_xgb_model = None
        
        # === WARM START: Load existing XGBoost model if available ===
        xgb_model_path = os.path.join(CFG.MODEL_DIR, f"xgb_{symbol}.json")
        if os.path.exists(xgb_model_path):
            try:
                loaded_model = xgb.XGBClassifier()
                loaded_model.load_model(xgb_model_path)
                # Evaluate loaded model to get baseline accuracy
                baseline_acc = (loaded_model.predict(X_val) == y_val).mean()
                best_xgb_acc = baseline_acc
                best_xgb_model = loaded_model
                self.xgb_model = loaded_model
                log.info(f"  🔥 Loaded existing XGBoost model | Baseline acc: {baseline_acc:.2%}")
            except Exception as e:
                log.warning(f"  Could not load XGBoost model: {e}")
        
        # Multiple training cycles for XGBoost with warm start
        for cycle in range(CFG.RETRAIN_CYCLES):
            if best_xgb_model is None:
                # First cycle with no existing model: train from scratch
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    eval_metric='logloss',
                    early_stopping_rounds=20,
                    random_state=42
                )
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                # Warm start: continue training from best model (loaded or from previous cycle)
                log.info(f"  🔥 XGB Warm start from best model (acc: {best_xgb_acc:.2%})")
                model = xgb.XGBClassifier(
                    n_estimators=100,  # Additional trees
                    max_depth=6,
                    learning_rate=0.01,  # Lower LR for fine-tuning
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    eval_metric='logloss',
                    early_stopping_rounds=20,
                    random_state=42 + cycle
                )
                # Use xgb_model parameter for warm start
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False,
                         xgb_model=best_xgb_model.get_booster())
            
            acc = (model.predict(X_val) == y_val).mean()
            log.info(f"  XGB Cycle {cycle+1}/{CFG.RETRAIN_CYCLES} | Accuracy: {acc:.2%}")
            
            if acc > best_xgb_acc:
                best_xgb_acc = acc
                best_xgb_model = model
                self.xgb_model = model
        
        log.info(f"✅ XGBoost Best Accuracy: {best_xgb_acc:.2%}")
        
        # Save best XGBoost model
        model_path = os.path.join(CFG.MODEL_DIR, f"xgb_{symbol}.json")
        try:
            self.xgb_model.save_model(model_path)
            log.info(f"💾 XGBoost model saved: {model_path}")
        except Exception as e:
            log.warning(f"Could not save XGBoost model: {e}")
        
        self.is_trained = True
        self.candles_processed = 0
        self.xgb_features = available_features  # Store for inference
    
    def should_retrain(self) -> bool:
        return self.candles_processed >= CFG.RETRAIN_INTERVAL


# ==================== POSITION SIZER (Unlimited Leverage Mode) ====================
class PositionSizer:
    """
    Aggressive position sizing for unlimited leverage.
    No margin constraints - pure risk-based sizing.
    """
    
    @staticmethod
    def calculate_lot(balance: float, atr_pips: float) -> float:
        """
        Calculate lot size based on risk and volatility.
        
        Formula: lot = (balance * risk_pct) / (SL_pips * USD_per_pip_per_lot)
        
        With volatility adjustment: If ATR > 2x normal, reduce size.
        """
        risk_amount = balance * CFG.RISK_PER_TRADE_PCT
        
        # Base calculation
        lot = risk_amount / (CFG.VIRTUAL_SL_PIPS * CFG.USD_PER_PIP_PER_LOT)
        
        # Volatility adjustment (reduce size in extreme volatility)
        if atr_pips > CFG.MIN_ATR_PIPS * 2:
            vol_factor = CFG.MIN_ATR_PIPS * 2 / atr_pips
            lot *= vol_factor
        
        # Apply limits
        lot = max(0.01, min(round(lot, 2), 10.0))  # Max 10 lots
        
        # Check max exposure
        max_lot = (balance * CFG.MAX_POSITION_PCT) / (CFG.VIRTUAL_SL_PIPS * CFG.USD_PER_PIP_PER_LOT)
        lot = min(lot, max_lot)
        
        return round(lot, 2)


# ==================== SESSION FILTER ====================
class SessionFilter:
    
    @staticmethod
    def is_active() -> bool:
        hour = datetime.utcnow().hour
        return (CFG.LONDON_START <= hour < CFG.LONDON_END or 
                CFG.NY_START <= hour < CFG.NY_END)
    
    @staticmethod
    def get_session() -> str:
        hour = datetime.utcnow().hour
        if CFG.LONDON_START <= hour < CFG.LONDON_END:
            return "LONDON"
        elif CFG.NY_START <= hour < CFG.NY_END:
            return "NEW_YORK"
        return "OFF"


# ==================== TRADE LOGGER ====================
class TradeLogger:
    
    def __init__(self):
        self.trades = []
        self.daily_pnl = 0.0
        self.wins = 0
        self.losses = 0
    
    def log_open(self, ticket: int, order_type: str, lot: float, price: float):
        self.trades.append({
            'time': datetime.utcnow().isoformat(),
            'ticket': ticket,
            'type': order_type,
            'lot': lot,
            'entry': price,
            'exit': None,
            'pnl': None,
            'reason': None
        })
    
    def log_close(self, ticket: int, exit_price: float, pnl: float, reason: str):
        for t in self.trades:
            if t['ticket'] == ticket and t['exit'] is None:
                t['exit'] = exit_price
                t['pnl'] = pnl
                t['reason'] = reason
                break
        
        self.daily_pnl += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        
        # Save to CSV
        df = pd.DataFrame([t for t in self.trades if t['exit'] is not None])
        df.to_csv(CFG.LOG_FILE, index=False)
    
    def get_stats(self) -> Dict:
        total = self.wins + self.losses
        return {
            'total': total,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': (self.wins / total * 100) if total > 0 else 0,
            'pnl': self.daily_pnl
        }
    
    def print_summary(self):
        s = self.get_stats()
        log.info("=" * 50)
        log.info(f"DAILY SUMMARY | Trades: {s['total']} | Win Rate: {s['win_rate']:.1f}% | P&L: ${s['pnl']:.2f}")
        log.info("=" * 50)


# ==================== MAIN BOT (The "Reflex") ====================
class StratosphereBot:
    """
    Main trading bot - The "Reflex" layer.
    Multi-symbol support with per-symbol ML engines.
    """
    
    def __init__(self):
        self.mt5 = MT5Handler()
        self.voms = VOMS()
        self.indicators = IndicatorEngine()
        self.logger = TradeLogger()
        self.running = False
        
        # Per-symbol state
        self.ml_engines: Dict[str, ThreadedMLEngine] = {}
        self.shared_signals: Dict[str, SharedSignal] = {}
        self.last_candle_times: Dict[str, any] = {}
    
    def initialize(self) -> bool:
        log.info("=" * 60)
        log.info("🚀 STRATOSPHERE BOT v2.0 - MULTI-SYMBOL")
        log.info("=" * 60)
        log.info(f"Symbols: {', '.join(CFG.SYMBOLS)}")
        log.info(f"Mode: {CFG.LEVERAGE_MODE} Leverage")
        log.info(f"Risk: {CFG.RISK_PER_TRADE_PCT:.1%} per trade")
        log.info(f"Prediction Horizon: {CFG.PREDICTION_HORIZON} candles")
        log.info(f"Training Cycles: {CFG.RETRAIN_CYCLES}")
        log.info("=" * 60)
        
        if not self.mt5.connect():
            return False
        
        # Initialize ML engines for each active symbol
        for symbol in self.mt5.active_symbols:
            log.info(f"📊 Initializing {symbol}...")
            
            df = self.mt5.fetch_candles(symbol, CFG.TRAIN_LOOKBACK)
            if len(df) < 200:
                log.warning(f"⚠️ {symbol}: Insufficient data ({len(df)} candles), skipping")
                continue
            
            df = self.indicators.calculate(df)
            
            # Create per-symbol ML engine
            # Use BTC-specific microstructure engine for Bitcoin
            shared_signal = SharedSignal()
            
            if 'BTC' in symbol.upper() and HAS_BTC_HFT:
                log.info(f"🔥 Using BTC Microstructure ML Engine for {symbol}")
                ml_engine = BTCMicrostructureMLEngine(shared_signal)
            else:
                ml_engine = ThreadedMLEngine(shared_signal)
            
            ml_engine.train(df, symbol)
            ml_engine.start()
            
            self.shared_signals[symbol] = shared_signal
            self.ml_engines[symbol] = ml_engine
            self.last_candle_times[symbol] = None
            
            log.info(f"✅ {symbol} ready")
        
        if not self.ml_engines:
            log.error("No symbols initialized!")
            return False
        
        log.info(f"✅ Initialization complete | {len(self.ml_engines)} symbols active")
        return True
    
    def run(self):
        """Main execution loop - multi-symbol tick processing"""
        if not self.initialize():
            return
        
        self.running = True
        tick_count = 0
        
        log.info("🟢 BOT RUNNING - Waiting for signals...")
        
        try:
            while self.running:
                try:
                    tick_start = time.perf_counter()
                    
                    # === REFLEX: Check VOMS exits for all positions ===
                    exits = self.voms.check_exits(self.mt5.get_tick)
                    for ticket, symbol, reason in exits:
                        if self.mt5.close_position(ticket, symbol, reason):
                            pos = self.voms.positions.get(ticket)
                            if pos:
                                bid, ask, _ = self.mt5.get_tick(symbol)
                                if pos.order_type == 'BUY':
                                    pnl = (bid - pos.entry_price) / CFG.PIP_VALUE * CFG.USD_PER_PIP_PER_LOT * pos.lot_size
                                else:
                                    pnl = (pos.entry_price - ask) / CFG.PIP_VALUE * CFG.USD_PER_PIP_PER_LOT * pos.lot_size
                                
                                self.logger.log_close(ticket, bid if pos.order_type == 'BUY' else ask, pnl, reason)
                            self.voms.remove_position(ticket)
                            self.voms.increment_api_calls()
                    
                    # === Process each symbol ===
                    for symbol in self.ml_engines.keys():
                        self._process_symbol(symbol)
                    
                    # === Status logging ===
                    tick_count += 1
                    if tick_count % 60000 == 0:
                        stats = self.logger.get_stats()
                        session = SessionFilter.get_session()
                        log.info(f"STATUS | Session: {session} | Trades: {stats['total']} | "
                                f"Win: {stats['win_rate']:.1f}% | P&L: ${stats['pnl']:.2f} | "
                                f"API: {self.voms.api_calls_today}/2000")
                    
                    # Sleep for tick interval
                    elapsed = time.perf_counter() - tick_start
                    sleep_time = max(0, CFG.TICK_SLEEP_MS / 1000 - elapsed)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    log.error(f"Loop error: {e}")
                    time.sleep(1)
        
        except KeyboardInterrupt:
            log.info("Stopped by user")
        
        finally:
            self.shutdown()
    
    def _process_symbol(self, symbol: str):
        """Process a single symbol - check for new candles and signals"""
        df = self.mt5.fetch_candles(symbol, 200)
        if len(df) == 0:
            return
        
        current_candle = df.index[-1]
        ml_engine = self.ml_engines.get(symbol)
        shared_signal = self.shared_signals.get(symbol)
        
        if not ml_engine or not shared_signal:
            return
        
        if self.last_candle_times.get(symbol) != current_candle:
            self.last_candle_times[symbol] = current_candle
            
            # Add indicators
            df = self.indicators.calculate(df)
            
            # Submit to ML engine (non-blocking)
            ml_engine.submit_data(df)
            
            # Check for retrain
            if ml_engine.should_retrain():
                train_df = self.mt5.fetch_candles(symbol, CFG.TRAIN_LOOKBACK)
                train_df = self.indicators.calculate(train_df)
                ml_engine.train(train_df, symbol)
        
        # === Check for entry signal ===
        if self.voms.get_position_count(symbol) == 0:  # One position per symbol
            if SessionFilter.is_active():
                signal, confidence, atr = shared_signal.read()
                
                if signal != SignalType.NONE and confidence >= CFG.ML_THRESHOLD:
                    self._execute_entry(symbol, signal, confidence, atr)
    
    def _execute_entry(self, symbol: str, signal: SignalType, confidence: float, atr: float):
        """Execute trade entry for symbol"""
        balance = self.mt5.get_balance()
        lot = PositionSizer.calculate_lot(balance, atr)
        
        order_type = "BUY" if signal == SignalType.LONG else "SELL"
        
        ticket = self.mt5.open_position(symbol, order_type, lot)
        
        if ticket:
            bid, ask, _ = self.mt5.get_tick(symbol)
            entry_price = ask if order_type == "BUY" else bid
            
            self.voms.add_position(ticket, symbol, order_type, entry_price, lot)
            self.voms.increment_api_calls()
            self.logger.log_open(ticket, order_type, lot, entry_price)
            self.shared_signals[symbol].clear()
            
            log.info(f"📈 {symbol} | {order_type} {lot} lots | Conf: {confidence:.1%} | ATR: {atr:.1f}")
    
    def shutdown(self):
        self.running = False
        for ml_engine in self.ml_engines.values():
            ml_engine.stop()
        self.logger.print_summary()
        self.mt5.disconnect()
        log.info("🔴 Bot shutdown complete")


# ==================== BACKTESTER ====================
class StratosphereBacktester:
    """Vectorized backtester with commission modeling"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
    
    def fetch_data(self, start: str = "2024-01-01") -> pd.DataFrame:
        try:
            import yfinance as yf
            log.info(f"Fetching data from {start}...")
            df = yf.download("GC=F", start=start, interval="1h")
            df.columns = [c.lower() for c in df.columns]
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            log.error(f"Data fetch failed: {e}")
            return pd.DataFrame()
    
    def run(self, df: pd.DataFrame) -> Dict:
        log.info("Running backtest...")
        
        df = IndicatorEngine.calculate(df)
        if len(df) < 500:
            return {}
        
        # Train on first half
        train_size = len(df) // 2
        train_df = df.iloc[:train_size]
        
        # Setup ML
        shared = SharedSignal()
        ml = ThreadedMLEngine(shared)
        ml.train(train_df)
        
        # Backtest
        balance = self.initial_balance
        position = None
        trades = []
        equity = [balance]
        
        for i in range(train_size + CFG.LSTM_TIMESTEPS, len(df)):
            current_df = df.iloc[:i+1]
            price = current_df['close'].iloc[-1]
            atr_pips = current_df['atr_pips'].iloc[-1] if 'atr_pips' in current_df.columns else 10
            
            # Get prediction
            ml._process_inference(current_df)
            signal, confidence, _ = shared.read()
            
            # Volatility gate
            if atr_pips < CFG.MIN_ATR_PIPS:
                signal = SignalType.NONE
            
            if position is None:
                # Entry
                if signal == SignalType.LONG and price > current_df['ema50'].iloc[-1]:
                    lot = PositionSizer.calculate_lot(balance, atr_pips)
                    position = {'type': 'BUY', 'entry': price, 'lot': lot,
                               'sl': price - CFG.VIRTUAL_SL_PIPS * CFG.PIP_VALUE,
                               'tp': price + CFG.VIRTUAL_TP_PIPS * CFG.PIP_VALUE}
                
                elif signal == SignalType.SHORT and price < current_df['ema50'].iloc[-1]:
                    lot = PositionSizer.calculate_lot(balance, atr_pips)
                    position = {'type': 'SELL', 'entry': price, 'lot': lot,
                               'sl': price + CFG.VIRTUAL_SL_PIPS * CFG.PIP_VALUE,
                               'tp': price - CFG.VIRTUAL_TP_PIPS * CFG.PIP_VALUE}
            else:
                # Exit check
                hit_sl = hit_tp = False
                
                if position['type'] == 'BUY':
                    hit_sl = price <= position['sl']
                    hit_tp = price >= position['tp']
                    pips = (price - position['entry']) / CFG.PIP_VALUE
                else:
                    hit_sl = price >= position['sl']
                    hit_tp = price <= position['tp']
                    pips = (position['entry'] - price) / CFG.PIP_VALUE
                
                if hit_sl or hit_tp:
                    # Calculate P&L with commission
                    gross_pnl = pips * CFG.USD_PER_PIP_PER_LOT * position['lot']
                    commission = CFG.COMMISSION_PER_LOT * position['lot']
                    net_pnl = gross_pnl - commission
                    
                    balance += net_pnl
                    trades.append({
                        'type': position['type'],
                        'pnl': net_pnl,
                        'pips': pips,
                        'exit': 'TP' if hit_tp else 'SL'
                    })
                    position = None
            
            equity.append(balance)
        
        # Calculate stats
        total = len(trades)
        wins = len([t for t in trades if t['pnl'] > 0])
        losses = total - wins
        
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        max_dd = 0
        peak = equity[0]
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            max_dd = max(max_dd, dd)
        
        returns = pd.Series(equity).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        results = {
            'initial': self.initial_balance,
            'final': round(balance, 2),
            'return_pct': round((balance - self.initial_balance) / self.initial_balance * 100, 2),
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': round(wins / total * 100, 2) if total > 0 else 0,
            'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            'max_drawdown': round(max_dd, 2),
            'sharpe': round(sharpe, 2)
        }
        
        return results
    
    def print_results(self, r: Dict):
        log.info("=" * 60)
        log.info("BACKTEST RESULTS")
        log.info("=" * 60)
        for k, v in r.items():
            log.info(f"{k}: {v}")
        log.info("=" * 60)


# ==================== SYMBOL SELECTION ====================
SYMBOL_MAP = {
    0: ["XAUUSDm", "BTCUSDm"],  # Both symbols
    1: ["XAUUSDm"],             # Gold only
    2: ["BTCUSDm"],             # Bitcoin only
}

# Runtime symbol list (mutable) - this gets updated by main()
ACTIVE_SYMBOLS: List[str] = ["XAUUSDm", "BTCUSDm"]


def get_active_symbols() -> List[str]:
    """Get the currently active symbols"""
    return ACTIVE_SYMBOLS


# ==================== MAIN ====================
def main():
    global ACTIVE_SYMBOLS
    
    parser = argparse.ArgumentParser(description='Stratosphere Multi-Symbol HFT Bot')
    parser.add_argument('--backtest', action='store_true')
    parser.add_argument('--risk', type=float, default=0.02)
    parser.add_argument('--start', type=str, default='2024-01-01')
    parser.add_argument('--symbol', '-s', type=int, default=None, choices=[0, 1, 2],
                        help='Symbol selection: 0=Both, 1=XAUUSDm (Gold), 2=BTCUSDm (Bitcoin)')
    args = parser.parse_args()
    
    # Interactive symbol selection if not provided via command line
    symbol_choice = args.symbol
    if symbol_choice is None:
        print("\n" + "=" * 50)
        print("  SYMBOL SELECTION")
        print("=" * 50)
        print("  [0] Both - XAUUSDm (Gold) + BTCUSDm (Bitcoin)")
        print("  [1] XAUUSDm - Gold Only")
        print("  [2] BTCUSDm - Bitcoin Only")
        print("=" * 50)
        
        while True:
            try:
                choice = input("  Enter your choice (0/1/2): ").strip()
                symbol_choice = int(choice)
                if symbol_choice in [0, 1, 2]:
                    break
                print("  ❌ Invalid choice. Please enter 0, 1, or 2.")
            except ValueError:
                print("  ❌ Invalid input. Please enter a number (0, 1, or 2).")
        print()
    
    # Update active symbols based on selection - MUST happen before bot init
    ACTIVE_SYMBOLS.clear()
    ACTIVE_SYMBOLS.extend(SYMBOL_MAP.get(symbol_choice, ["XAUUSDm", "BTCUSDm"]))
    
    symbols_str = ", ".join(ACTIVE_SYMBOLS)
    mode_name = {0: "MULTI-SYMBOL", 1: "GOLD ONLY", 2: "BITCOIN ONLY"}.get(symbol_choice, "MULTI")
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║        STRATOSPHERE - {mode_name:<15} HFT BOT v2.1          ║
    ║   VOMS + Threaded ML + Volatility Gating + Unlimited Leverage ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║   Symbol Selection: 0=Both | 1=Gold | 2=Bitcoin               ║
    ║   Active: {symbols_str:<53} ║
    ║   Training Cycles: {CFG.RETRAIN_CYCLES} | Prediction Horizon: {CFG.PREDICTION_HORIZON} candles      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"DEBUG: ACTIVE_SYMBOLS = {ACTIVE_SYMBOLS}")  # Debug line
    
    CFG.RISK_PER_TRADE_PCT = args.risk
    
    if args.backtest:
        bt = StratosphereBacktester()
        df = bt.fetch_data(args.start)
        if len(df) > 0:
            results = bt.run(df)
            bt.print_results(results)
    else:
        log.warning("⚠️ LIVE TRADING MODE - Use at your own risk!")
        bot = StratosphereBot()
        bot.run()


if __name__ == "__main__":
    main()
