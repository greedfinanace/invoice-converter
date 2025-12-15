"""
================================================================================
STRATOSPHERE v3.0 - BTCUSDm 5-MINUTE MOMENTUM FEATURES
================================================================================
BTCUSDm EXCLUSIVE - Momentum & Volatility Expansion

┌─────────────────────────────────────────────────────────────────────┐
│ CORE SPECIFICATION                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Base candle feed      : 1-minute (raw data)                         │
│ Effective timeframe   : 5-minute (rolling aggregation)              │
│ Prediction horizon    : 3 candles = ~15 minutes forward             │
└─────────────────────────────────────────────────────────────────────┘

Feature Categories:
- Returns (log returns over 1, 3, 5 periods)
- Volatility (ATR, std, expansion ratio)
- Candle structure (wick ratios, body %, range expansion)
- Momentum (3-period, 5-period, strength)
- Volume (delta, acceleration, surge)
- Regime flags (low/high/extreme volatility)
- Session encoding (Asia/London/NY)

CLEAN + ORTHOGONAL features only. No lagging indicators.
All features z-score normalized with correlation pruning (>85% → drop).
================================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def aggregate_to_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute candles to 5-minute candles.
    This is the EFFECTIVE TIMEFRAME for BTC momentum mode.
    """
    df = df_1m.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            df.index = pd.to_datetime(df['time'])
        else:
            # Create dummy datetime index
            df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
    
    # Resample to 5-minute
    df_5m = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_5m


class SpreadAwareFeatures:
    """
    BTC 5-Minute Momentum Mode Feature Engineering.
    
    You look at 1-minute data
    You aggregate structure over 5 minutes
    You predict ~15 minutes ahead (3 x 5-min candles)
    
    Core Features:
    - Log returns (1m, 3m, 5m on 5-min bars)
    - Rolling volatility (ATR / std)
    - Wick-to-body ratios
    - Candle range expansion
    - EMA deviation (fast only)
    - Volume delta (tick-based)
    - Volume acceleration
    - Volatility regime flags
    - Session encoding (Asia / London / NY)
    
    5-Minute Aggregated Features:
    - 5-minute aggregated returns
    - 5-minute high/low range
    - Volatility expansion ratio (ATR5 / ATR20)
    """
    
    # Core features for both assets
    CORE_FEATURES = [
        # Returns
        'log_ret_1', 'log_ret_3', 'log_ret_5',
        # Volatility
        'atr_5', 'atr_10', 'ret_std_5', 'ret_std_10',
        'vol_ratio',  # ATR5/ATR10
        # Candle structure
        'wick_body_ratio', 'upper_wick_pct', 'lower_wick_pct',
        'body_pct', 'range_expansion',
        # EMA deviation
        'ema_6_dev', 'ema_12_dev',
        # Volume
        'vol_delta', 'vol_delta_3', 'vol_accel', 'vol_surge',
        # Regime
        'vol_regime_low', 'vol_regime_high', 'vol_regime_extreme',
        # Session
        'session_asia', 'session_london', 'session_ny',
        # Price position
        'close_position', 'range_norm'
    ]
    
    # BTC-only features (momentum & volatility expansion focused)
    BTC_FEATURES = [
        'ret_5m_agg', 'range_5m', 'vol_expansion_ratio',
        # Momentum features
        'momentum_3', 'momentum_5', 'momentum_strength',
        # Volatility expansion
        'vol_expansion_flag', 'atr_acceleration'
    ]
    
    def __init__(self, symbol: str = "BTC", correlation_threshold: float = 0.85):
        self.symbol = symbol.upper()
        self.is_btc = "BTC" in self.symbol
        self.correlation_threshold = correlation_threshold
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.dropped_features: List[str] = []
        self._fitted = False
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features for the given OHLCV data."""
        df = df.copy()
        
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']
        volume = df['volume']
        
        # ==================== LOG RETURNS ====================
        df['log_ret_1'] = np.log(close / close.shift(1))
        df['log_ret_3'] = np.log(close / close.shift(3))
        df['log_ret_5'] = np.log(close / close.shift(5))
        
        # ==================== VOLATILITY (ATR / STD) ====================
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        df['atr_5'] = tr.rolling(5).mean()
        df['atr_10'] = tr.rolling(10).mean()
        df['atr_20'] = tr.rolling(20).mean()
        
        df['ret_std_5'] = df['log_ret_1'].rolling(5).std()
        df['ret_std_10'] = df['log_ret_1'].rolling(10).std()
        
        df['vol_ratio'] = df['atr_5'] / df['atr_10'].replace(0, np.nan)
        
        # ==================== CANDLE STRUCTURE ====================
        candle_range = high - low
        candle_body = (close - open_).abs()
        
        upper_wick = high - np.maximum(close, open_)
        lower_wick = np.minimum(close, open_) - low
        
        df['wick_body_ratio'] = (upper_wick + lower_wick) / candle_body.replace(0, np.nan)
        df['upper_wick_pct'] = upper_wick / candle_range.replace(0, np.nan)
        df['lower_wick_pct'] = lower_wick / candle_range.replace(0, np.nan)
        df['body_pct'] = candle_body / candle_range.replace(0, np.nan)
        
        # Range expansion (current range vs rolling average)
        df['range_expansion'] = candle_range / candle_range.rolling(10).mean().replace(0, np.nan)
        
        # ==================== EMA DEVIATION (FAST ONLY) ====================
        df['ema_6'] = close.ewm(span=6, adjust=False).mean()
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        
        df['ema_6_dev'] = (close - df['ema_6']) / df['ema_6']
        df['ema_12_dev'] = (close - df['ema_12']) / df['ema_12']
        
        # ==================== VOLUME FEATURES ====================
        df['vol_delta'] = volume.diff(1)
        df['vol_delta_3'] = volume.diff(3)
        
        vol_ma = volume.rolling(10).mean()
        df['vol_accel'] = df['vol_delta'].diff(1)  # Second derivative
        df['vol_surge'] = volume / vol_ma.replace(0, np.nan)
        
        # ==================== VOLATILITY REGIME FLAGS ====================
        vol_percentile = df['atr_5'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        df['vol_regime_low'] = (vol_percentile < 0.25).astype(int)
        df['vol_regime_high'] = ((vol_percentile >= 0.75) & (vol_percentile < 0.95)).astype(int)
        df['vol_regime_extreme'] = (vol_percentile >= 0.95).astype(int)
        
        # ==================== SESSION ENCODING ====================
        if 'time' in df.columns:
            hour = pd.to_datetime(df['time']).dt.hour
        elif df.index.dtype == 'datetime64[ns]':
            hour = df.index.hour
        else:
            # Assume UTC, create dummy session based on index position
            hour = pd.Series(range(len(df))) % 24
        
        df['session_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
        df['session_london'] = ((hour >= 8) & (hour < 13)).astype(int)
        df['session_ny'] = ((hour >= 13) & (hour < 21)).astype(int)
        
        # ==================== PRICE POSITION ====================
        df['close_position'] = (close - low) / candle_range.replace(0, np.nan)
        df['range_norm'] = candle_range / close
        
        # ==================== BTC-ONLY FEATURES (MOMENTUM & EXPANSION) ====================
        if self.is_btc:
            # 5-minute aggregated returns
            df['ret_5m_agg'] = np.log(close / close.shift(5))
            
            # 5-minute high/low range
            df['range_5m'] = high.rolling(5).max() - low.rolling(5).min()
            
            # Volatility expansion ratio (key for momentum detection)
            df['vol_expansion_ratio'] = df['atr_5'] / df['atr_20'].replace(0, np.nan)
            
            # === MOMENTUM FEATURES ===
            # Price momentum over 3 and 5 periods
            df['momentum_3'] = close - close.shift(3)
            df['momentum_5'] = close - close.shift(5)
            
            # Momentum strength (normalized by ATR)
            df['momentum_strength'] = df['momentum_5'].abs() / df['atr_5'].replace(0, np.nan)
            
            # === VOLATILITY EXPANSION FEATURES ===
            # Flag when volatility is expanding (ATR5 > ATR20)
            df['vol_expansion_flag'] = (df['vol_expansion_ratio'] > 1.0).astype(int)
            
            # ATR acceleration (rate of change of ATR)
            df['atr_acceleration'] = df['atr_5'].diff(3) / df['atr_5'].shift(3).replace(0, np.nan)
        
        return df.replace([np.inf, -np.inf], np.nan)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for this asset."""
        features = self.CORE_FEATURES.copy()
        if self.is_btc:
            features.extend(self.BTC_FEATURES)
        return features
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate features, prune correlated ones, and z-score normalize.
        Returns (X_scaled, feature_names).
        """
        df_feat = self.calculate(df)
        
        # Get available features
        all_features = self.get_feature_names()
        available = [f for f in all_features if f in df_feat.columns]
        
        # Extract feature matrix
        X = df_feat[available].dropna()
        
        if len(X) < 100:
            raise ValueError(f"Insufficient data after feature calculation: {len(X)} rows")
        
        # Correlation pruning
        X, kept_features = self._prune_correlated(X, available)
        
        # Z-score normalization
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = kept_features
        self._fitted = True
        
        return X_scaled, kept_features
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scaler."""
        if not self._fitted:
            raise ValueError("Must call fit_transform first")
        
        df_feat = self.calculate(df)
        X = df_feat[self.feature_names].values
        
        # Handle NaN by forward fill then backward fill
        X = pd.DataFrame(X).ffill().bfill().values
        
        return self.scaler.transform(X)
    
    def _prune_correlated(self, X: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with correlation > threshold."""
        corr_matrix = X.corr().abs()
        
        # Upper triangle mask
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
        
        self.dropped_features = to_drop
        kept = [f for f in features if f not in to_drop]
        
        return X[kept], kept
    
    def get_regime(self, df: pd.DataFrame) -> Dict:
        """
        Classify current volatility regime with expansion/contraction detection.
        Returns dict with regime info for filtering (v3.1 enhanced).
        """
        df_feat = self.calculate(df)
        
        if len(df_feat) < 25:
            return {'regime': 'medium', 'atr_ratio': 1.0, 'tradeable': True, 
                    'is_expanding': False, 'is_contracting': False}
        
        last = df_feat.iloc[-1]
        
        vol_ratio = last.get('vol_ratio', 1.0)
        atr_5 = last.get('atr_5', 0)
        
        # Get volatility expansion ratio (ATR5/ATR20) for BTC
        vol_expansion_ratio = last.get('vol_expansion_ratio', 1.0) if self.is_btc else vol_ratio
        
        # Classify regime
        if last.get('vol_regime_extreme', 0) == 1:
            regime = 'extreme'
        elif last.get('vol_regime_high', 0) == 1:
            regime = 'high'
        elif last.get('vol_regime_low', 0) == 1:
            regime = 'low'
        else:
            regime = 'medium'
        
        # Volatility expansion/contraction detection (v3.1)
        # Use vol_expansion_ratio (ATR5/ATR20) for more accurate detection
        atr_ratio_val = float(vol_expansion_ratio) if not np.isnan(vol_expansion_ratio) else 1.0
        is_expanding = atr_ratio_val > 1.15
        is_contracting = atr_ratio_val < 0.85
        
        # Tradeable check (v3.1: only skip severe compression, not all low vol)
        # Severe compression = low regime AND atr_ratio < 0.7
        tradeable = not (regime == 'low' and atr_ratio_val < 0.7)
        
        return {
            'regime': regime,
            'atr_ratio': atr_ratio_val,
            'atr_5': float(atr_5) if not np.isnan(atr_5) else 0,
            'tradeable': tradeable,
            'is_expanding': is_expanding,
            'is_contracting': is_contracting,
            'vol_expansion_ratio': atr_ratio_val
        }
    
    def get_session(self, df: pd.DataFrame) -> str:
        """Get current trading session."""
        df_feat = self.calculate(df)
        last = df_feat.iloc[-1]
        
        if last.get('session_london', 0) == 1:
            return 'london'
        elif last.get('session_ny', 0) == 1:
            return 'ny'
        elif last.get('session_asia', 0) == 1:
            return 'asia'
        return 'other'
