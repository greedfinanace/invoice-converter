"""
================================================================================
BTC HFT MICROSTRUCTURE PREDICTOR v3.0 - OPTIMIZED TCN + LightGBM
================================================================================
Ultra-optimized 1-3 candle prediction for BTC/XAU HFT

OPTIMIZATION LAYERS:
1. Dynamic Confidence Thresholds (0.55-0.62 based on volatility)
2. Volatility Regime Filtering (ATR ratios, skip low-vol)
3. Session-Based Trade Gating (UTC windows)
4. Optimized TCN (dilation, dropout, residuals)
5. Optimized LightGBM (feature bagging, regularization)
6. Orthogonal Microstructure Features Only
7. Trade Frequency Optimization (margin_min filtering)
8. Adaptive TP/SL Logic

Target: Higher win rate, higher profitability, cleaner trade flow
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime, timezone
import joblib
import warnings
warnings.filterwarnings('ignore')

# LightGBM
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# TensorFlow/Keras for TCN
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv1D, Dense, Dropout, BatchNormalization,
        Add, Activation, GlobalAveragePooling1D, SpatialDropout1D
    )
    from tensorflow.keras.regularizers import l2
    HAS_TF = True
except ImportError:
    HAS_TF = False


# ==================== VOLATILITY REGIME CLASSIFIER ====================
class VolatilityRegime:
    """
    Classify market volatility regime for trade filtering.
    Regimes: LOW, MEDIUM, HIGH, EXTREME
    """
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"
    
    @staticmethod
    def classify(df: pd.DataFrame) -> dict:
        """
        Classify current volatility regime using multiple indicators.
        Returns regime and metrics.
        """
        if len(df) < 25:
            return {'regime': VolatilityRegime.MEDIUM, 'atr_ratio': 1.0, 'vol_percentile': 50}
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR(5) / ATR(20) ratio
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr_5 = tr.rolling(5).mean().iloc[-1]
        atr_20 = tr.rolling(20).mean().iloc[-1]
        atr_ratio = atr_5 / atr_20 if atr_20 > 0 else 1.0
        
        # Candle volatility (range / close)
        candle_vol = ((high - low) / close).rolling(10).mean().iloc[-1]
        
        # Historical volatility percentile
        returns = close.pct_change().dropna()
        current_vol = returns.rolling(5).std().iloc[-1]
        vol_20 = returns.rolling(20).std().iloc[-1]
        vol_ratio = current_vol / vol_20 if vol_20 > 0 else 1.0
        
        # Classify regime
        if atr_ratio < 0.7 or vol_ratio < 0.6:
            regime = VolatilityRegime.LOW
        elif atr_ratio > 1.8 or vol_ratio > 2.0:
            regime = VolatilityRegime.EXTREME
        elif atr_ratio > 1.3 or vol_ratio > 1.4:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.MEDIUM
        
        return {
            'regime': regime,
            'atr_ratio': round(atr_ratio, 3),
            'vol_ratio': round(vol_ratio, 3),
            'candle_vol': round(candle_vol, 6),
            'atr_5': atr_5
        }


# ==================== SESSION FILTER ====================
class SessionFilter:
    """
    Session-based trade gating for BTC and XAU.
    Filters low-liquidity windows.
    """
    
    # BTC optimal windows (UTC)
    BTC_FULL_SPEED = [(12, 18)]  # 12:00-18:00 UTC - highest volume
    BTC_REDUCED = [(0, 5)]       # 00:00-05:00 UTC - lower quality
    
    # XAU optimal windows (UTC)
    XAU_FULL_SPEED = [(7, 11), (13, 17)]  # London + NY overlap
    XAU_REDUCED = [(22, 6)]               # Asia - lower accuracy
    
    @staticmethod
    def get_session_multiplier(symbol: str = "BTC") -> float:
        """
        Get threshold multiplier based on current session.
        Returns: 1.0 = normal, >1.0 = stricter, <1.0 = looser
        """
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        if "BTC" in symbol.upper():
            # Weekend penalty for BTC
            if weekday >= 5:
                return 1.08  # Stricter on weekends
            
            # Full speed windows
            for start, end in SessionFilter.BTC_FULL_SPEED:
                if start <= hour < end:
                    return 0.97  # Slightly looser
            
            # Reduced windows
            for start, end in SessionFilter.BTC_REDUCED:
                if start <= hour or hour < end:
                    return 1.10  # Stricter
            
            return 1.0
        
        else:  # XAU/Gold
            # Full speed windows
            for start, end in SessionFilter.XAU_FULL_SPEED:
                if start <= hour < end:
                    return 0.95  # Looser during London/NY
            
            # Asia session
            if hour >= 22 or hour < 6:
                return 1.12  # Much stricter
            
            return 1.0
    
    @staticmethod
    def is_optimal_session(symbol: str = "BTC") -> bool:
        """Check if current session is optimal for trading."""
        multiplier = SessionFilter.get_session_multiplier(symbol)
        return multiplier <= 1.0



# ==================== OPTIMIZED MICROSTRUCTURE FEATURES ====================
class BTCMicrostructureFeatures:
    """
    Orthogonal microstructure features only.
    Removed: slow indicators, redundant momentum, high-correlation features.
    Kept: returns, volatility ratios, orderflow deltas, wick/body, EMA deviations
    """
    
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate optimized feature set."""
        df = df.copy()
        
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']
        volume = df['volume']
        
        # ==================== RETURNS (Core) ====================
        df['ret_1'] = close.pct_change(1)
        df['ret_3'] = close.pct_change(3)
        df['ret_5'] = close.pct_change(5)
        
        # ==================== VOLATILITY RATIOS ====================
        df['ret_vol_5'] = df['ret_1'].rolling(5).std()
        df['ret_vol_10'] = df['ret_1'].rolling(10).std()
        df['vol_ratio'] = df['ret_vol_5'] / df['ret_vol_10'].replace(0, np.nan)
        
        # ATR-based volatility
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        df['atr_5'] = tr.rolling(5).mean()
        df['atr_20'] = tr.rolling(20).mean()
        df['atr_ratio'] = df['atr_5'] / df['atr_20'].replace(0, np.nan)
        
        # ==================== WICK/BODY RATIOS ====================
        candle_range = high - low
        candle_body = (close - open_).abs()
        df['body_ratio'] = candle_body / candle_range.replace(0, np.nan)
        
        upper_wick = high - np.maximum(close, open_)
        lower_wick = np.minimum(close, open_) - low
        df['wick_ratio'] = (upper_wick + lower_wick) / candle_body.replace(0, np.nan)
        df['upper_wick_ratio'] = upper_wick / candle_range.replace(0, np.nan)
        df['lower_wick_ratio'] = lower_wick / candle_range.replace(0, np.nan)
        
        # ==================== ORDERFLOW DELTAS ====================
        df['vol_delta'] = volume.diff(1)
        df['vol_delta_3'] = volume.diff(3)
        vol_ma = volume.rolling(10).mean()
        df['vol_surge'] = volume / vol_ma.replace(0, np.nan)
        df['vol_spike'] = (volume > 2 * vol_ma).astype(int)
        
        # ==================== EMA DEVIATIONS ====================
        df['ema_6'] = close.ewm(span=6, adjust=False).mean()
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_6_dev'] = (close - df['ema_6']) / df['ema_6']
        df['ema_12_dev'] = (close - df['ema_12']) / df['ema_12']
        df['ema_slope'] = df['ema_6'].diff(1) / df['ema_6'].shift(1)
        
        # ==================== MICRO VWAP ====================
        typical_price = (high + low + close) / 3
        df['vwap_5'] = (typical_price * volume).rolling(5).sum() / volume.rolling(5).sum()
        df['vwap_dev'] = (close - df['vwap_5']) / df['vwap_5']
        
        # ==================== PRICE STRUCTURE ====================
        df['close_position'] = (close - low) / candle_range.replace(0, np.nan)
        df['candle_dir'] = np.sign(close - open_)
        df['range_norm'] = candle_range / close
        
        # Consolidation detection (for skipping)
        df['range_ma'] = candle_range.rolling(10).mean()
        df['is_consolidation'] = (candle_range < df['range_ma'] * 0.5).astype(int)
        
        return df.replace([np.inf, -np.inf], np.nan).dropna()
    
    @staticmethod
    def get_feature_names() -> list:
        """Orthogonal features only - removed redundant/correlated."""
        return [
            # Returns
            'ret_1', 'ret_3', 'ret_5',
            # Volatility ratios
            'ret_vol_5', 'vol_ratio', 'atr_ratio',
            # Wick/body
            'body_ratio', 'wick_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
            # Orderflow
            'vol_delta', 'vol_surge', 'vol_spike',
            # EMA deviations
            'ema_6_dev', 'ema_12_dev', 'ema_slope',
            # VWAP
            'vwap_dev',
            # Structure
            'close_position', 'candle_dir', 'range_norm'
        ]


# ==================== OPTIMIZED TCN ====================
class OptimizedTCNBlock(keras.layers.Layer):
    """
    Optimized TCN block with:
    - Increased dilation for longer patterns
    - Spatial dropout for regularization
    - L2 regularization
    - Residual connections
    """
    
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.conv1 = Conv1D(
            self.filters, self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='causal',
            kernel_regularizer=l2(0.001),
            activation=None
        )
        self.bn1 = BatchNormalization()
        self.dropout1 = SpatialDropout1D(self.dropout_rate)
        
        self.conv2 = Conv1D(
            self.filters, self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='causal',
            kernel_regularizer=l2(0.001),
            activation=None
        )
        self.bn2 = BatchNormalization()
        self.dropout2 = SpatialDropout1D(self.dropout_rate)
        
        if input_shape[-1] != self.filters:
            self.residual_conv = Conv1D(self.filters, 1, padding='same')
        else:
            self.residual_conv = None
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = Activation('relu')(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = Activation('relu')(x)
        x = self.dropout2(x, training=training)
        
        if self.residual_conv is not None:
            residual = self.residual_conv(inputs)
        else:
            residual = inputs
        
        return Add()([x, residual])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate
        })
        return config


def build_optimized_tcn(input_shape, num_filters=24, kernel_size=3, num_blocks=4, dropout=0.18):
    """
    Optimized TCN for chaotic BTC and structured XAU.
    
    Optimizations:
    - Reduced filters (24 vs 32) to prevent overfit
    - Increased blocks (4) with higher dilations
    - Spatial dropout (0.18) for regularization
    - L2 regularization on conv layers
    """
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Stack TCN blocks with increasing dilation: 1, 2, 4, 8
    for i in range(num_blocks):
        dilation_rate = 2 ** i
        x = OptimizedTCNBlock(num_filters, kernel_size, dilation_rate, dropout)(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(12, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(dropout)(x)
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0008),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model



# ==================== DYNAMIC THRESHOLD CALCULATOR ====================
class DynamicThreshold:
    """
    Dynamic confidence threshold based on volatility regime.
    Base: 0.55, Range: 0.53-0.62
    """
    
    BASE_THRESHOLD = 0.55
    MIN_THRESHOLD = 0.53
    MAX_THRESHOLD = 0.62
    
    # Margin minimum for flat prediction filtering
    MARGIN_MIN = 0.07  # Skip if |prob - 0.5| < 0.07
    
    @staticmethod
    def calculate(vol_regime: dict, symbol: str = "BTC") -> float:
        """
        Calculate dynamic threshold based on volatility and session.
        
        Low vol → higher threshold (stricter)
        High vol → lower threshold (more trades)
        Extreme vol → higher threshold (protect capital)
        """
        regime = vol_regime.get('regime', VolatilityRegime.MEDIUM)
        atr_ratio = vol_regime.get('atr_ratio', 1.0)
        
        # Base adjustment by regime
        if regime == VolatilityRegime.LOW:
            threshold = DynamicThreshold.BASE_THRESHOLD + 0.05  # 0.60
        elif regime == VolatilityRegime.HIGH:
            threshold = DynamicThreshold.BASE_THRESHOLD - 0.02  # 0.53
        elif regime == VolatilityRegime.EXTREME:
            threshold = DynamicThreshold.BASE_THRESHOLD + 0.03  # 0.58
        else:  # MEDIUM
            threshold = DynamicThreshold.BASE_THRESHOLD  # 0.55
        
        # Fine-tune with ATR ratio
        vol_adjustment = (atr_ratio - 1.0) * 0.03
        threshold -= vol_adjustment  # Higher ATR = lower threshold
        
        # Session multiplier
        session_mult = SessionFilter.get_session_multiplier(symbol)
        threshold *= session_mult
        
        # Clamp to range
        threshold = max(DynamicThreshold.MIN_THRESHOLD, 
                       min(DynamicThreshold.MAX_THRESHOLD, threshold))
        
        return round(threshold, 3)
    
    @staticmethod
    def should_skip_flat_prediction(prob: float) -> bool:
        """Skip trades where prediction is too close to 0.5."""
        return abs(prob - 0.5) < DynamicThreshold.MARGIN_MIN


# ==================== ADAPTIVE TP/SL CALCULATOR ====================
class AdaptiveTPSL:
    """
    Dynamic TP/SL based on ATR and candle structure.
    Maximizes average trade expectancy.
    """
    
    @staticmethod
    def calculate(atr: float, vol_regime: dict, symbol: str = "BTC") -> dict:
        """
        Calculate adaptive TP/SL levels.
        
        Returns dict with:
        - sl_pips: Stop loss in price units
        - tp_pips: Take profit in price units
        - partial_tp: Partial close level
        - breakeven_level: Move SL to breakeven after this
        """
        regime = vol_regime.get('regime', VolatilityRegime.MEDIUM)
        
        # Base multipliers
        if regime == VolatilityRegime.LOW:
            sl_mult = 1.2
            tp_mult = 1.8
        elif regime == VolatilityRegime.HIGH:
            sl_mult = 1.5
            tp_mult = 2.5
        elif regime == VolatilityRegime.EXTREME:
            sl_mult = 2.0
            tp_mult = 3.0
        else:  # MEDIUM
            sl_mult = 1.3
            tp_mult = 2.2
        
        # Calculate levels
        sl_pips = atr * sl_mult
        tp_pips = atr * tp_mult
        
        # Partial close at 50% of TP
        partial_tp = tp_pips * 0.5
        
        # Breakeven after 40% of TP
        breakeven_level = tp_pips * 0.4
        
        return {
            'sl_pips': round(sl_pips, 2),
            'tp_pips': round(tp_pips, 2),
            'partial_tp': round(partial_tp, 2),
            'breakeven_level': round(breakeven_level, 2),
            'risk_reward': round(tp_pips / sl_pips, 2) if sl_pips > 0 else 2.0
        }


# ==================== MAIN PREDICTOR CLASS ====================
class BTCHFTPredictor:
    """
    Optimized TCN + LightGBM ensemble for BTC/XAU HFT.
    
    Optimizations:
    1. Dynamic confidence thresholds (0.53-0.62)
    2. Volatility regime filtering
    3. Session-based trade gating
    4. Optimized TCN architecture
    5. Optimized LightGBM hyperparameters
    6. Orthogonal microstructure features
    7. Flat prediction filtering (margin_min)
    8. Adaptive TP/SL
    """
    
    def __init__(self, 
                 prediction_horizon: int = 2,
                 base_threshold: float = 0.55,
                 use_tcn: bool = True,
                 tcn_timesteps: int = 30,  # Shorter sequences
                 symbol: str = "BTC"):
        
        self.prediction_horizon = prediction_horizon
        self.base_threshold = base_threshold
        self.use_tcn = use_tcn and HAS_TF
        self.tcn_timesteps = tcn_timesteps
        self.symbol = symbol
        
        self.features = BTCMicrostructureFeatures.get_feature_names()
        self.scaler = StandardScaler()
        
        self.lgbm_model = None
        self.tcn_model = None
        self.is_trained = False
        
        # Stats tracking
        self.trades_filtered = 0
        self.trades_passed = 0
        
    def _prepare_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Create targets for prediction."""
        future_close = df['close'].shift(-self.prediction_horizon)
        current_close = df['close']
        targets = (future_close > current_close).astype(int)
        return targets.values
    
    def _prepare_tcn_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Prepare sequences for TCN."""
        X_seq, y_seq = [], []
        for i in range(self.tcn_timesteps, len(X)):
            X_seq.append(X[i - self.tcn_timesteps:i])
            y_seq.append([1 - y[i], y[i]])
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, df: pd.DataFrame, retrain_cycles: int = 3) -> dict:
        """
        Train optimized TCN + LightGBM ensemble.
        """
        df_feat = BTCMicrostructureFeatures.calculate(df)
        
        if len(df_feat) < 200:
            raise ValueError(f"Insufficient data: {len(df_feat)} rows")
        
        X = df_feat[self.features].values[:-self.prediction_horizon]
        y = self._prepare_targets(df_feat)[:-self.prediction_horizon]
        
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask].astype(int)
        
        X_scaled = self.scaler.fit_transform(X)
        
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # ==================== OPTIMIZED LIGHTGBM ====================
        best_lgbm_acc = 0
        best_lgbm_model = None
        
        if HAS_LGBM:
            for cycle in range(retrain_cycles):
                random_seed = np.random.randint(0, 10000)
                
                model = lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=3,              # Reduced to prevent overfit
                    learning_rate=0.05,       # Tuned: 0.03-0.07
                    num_leaves=12,            # Reduced
                    subsample=0.75,
                    colsample_bytree=0.75,
                    feature_fraction=0.8,     # Feature bagging
                    bagging_fraction=0.8,     # Row bagging
                    bagging_freq=5,
                    min_child_weight=10,      # Filter noise
                    min_child_samples=20,
                    reg_alpha=0.1,            # L1 regularization
                    reg_lambda=0.1,           # L2 regularization
                    class_weight='balanced',
                    random_state=random_seed,
                    verbose=-1,
                    n_jobs=-1
                )
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(15, verbose=False)]
                )
                
                acc = (model.predict(X_val) == y_val).mean()
                if acc > best_lgbm_acc:
                    best_lgbm_acc = acc
                    best_lgbm_model = model
            
            self.lgbm_model = best_lgbm_model
        
        # ==================== OPTIMIZED TCN ====================
        best_tcn_acc = 0
        
        if self.use_tcn:
            X_tcn_train, y_tcn_train = self._prepare_tcn_sequences(X_train, y_train)
            X_tcn_val, y_tcn_val = self._prepare_tcn_sequences(X_val, y_val)
            
            if len(X_tcn_train) > 50:
                for cycle in range(retrain_cycles):
                    keras.backend.clear_session()
                    
                    model = build_optimized_tcn(
                        input_shape=(self.tcn_timesteps, len(self.features)),
                        num_filters=24,
                        kernel_size=3,
                        num_blocks=4,
                        dropout=0.18
                    )
                    
                    early_stop = keras.callbacks.EarlyStopping(
                        monitor='val_accuracy',
                        patience=8,
                        restore_best_weights=True
                    )
                    
                    reduce_lr = keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=4,
                        min_lr=0.0001
                    )
                    
                    tcn_class_weights = {0: class_weight_dict[0], 1: class_weight_dict[1]}
                    
                    model.fit(
                        X_tcn_train, y_tcn_train,
                        validation_data=(X_tcn_val, y_tcn_val),
                        epochs=40,
                        batch_size=32,
                        class_weight=tcn_class_weights,
                        callbacks=[early_stop, reduce_lr],
                        verbose=0
                    )
                    
                    acc = model.evaluate(X_tcn_val, y_tcn_val, verbose=0)[1]
                    if acc > best_tcn_acc:
                        best_tcn_acc = acc
                        self.tcn_model = model
        
        self.is_trained = True
        
        return {
            'lgbm_accuracy': round(best_lgbm_acc, 4),
            'tcn_accuracy': round(best_tcn_acc, 4) if self.use_tcn else None,
            'samples_train': len(y_train),
            'samples_val': len(y_val),
            'features': len(self.features),
            'prediction_horizon': self.prediction_horizon
        }

    
    def predict(self, df: pd.DataFrame) -> dict:
        """
        Get prediction with all optimization layers applied.
        
        Filters:
        1. Volatility regime check
        2. Session check
        3. Dynamic threshold
        4. Flat prediction filter
        5. Consolidation filter
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        df_feat = BTCMicrostructureFeatures.calculate(df)
        
        if len(df_feat) < self.tcn_timesteps + 1:
            return self._hold_result("insufficient_data")
        
        # ==================== VOLATILITY REGIME ====================
        vol_regime = VolatilityRegime.classify(df_feat)
        
        # Skip low volatility (no edge)
        if vol_regime['regime'] == VolatilityRegime.LOW:
            self.trades_filtered += 1
            return self._hold_result("low_volatility", vol_regime)
        
        # ==================== CONSOLIDATION CHECK ====================
        if df_feat['is_consolidation'].iloc[-1] == 1:
            self.trades_filtered += 1
            return self._hold_result("consolidation", vol_regime)
        
        # ==================== DYNAMIC THRESHOLD ====================
        threshold = DynamicThreshold.calculate(vol_regime, self.symbol)
        
        # ==================== MODEL PREDICTIONS ====================
        X = df_feat[self.features].iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        # LightGBM
        lgbm_up_prob = 0.5
        if self.lgbm_model is not None:
            lgbm_proba = self.lgbm_model.predict_proba(X_scaled)[0]
            lgbm_up_prob = lgbm_proba[1]
        
        # TCN
        tcn_up_prob = 0.5
        if self.use_tcn and self.tcn_model is not None:
            X_tcn = df_feat[self.features].iloc[-self.tcn_timesteps:].values
            X_tcn_scaled = self.scaler.transform(X_tcn)
            X_tcn_seq = X_tcn_scaled.reshape(1, self.tcn_timesteps, len(self.features))
            tcn_proba = self.tcn_model.predict(X_tcn_seq, verbose=0)[0]
            tcn_up_prob = tcn_proba[1]
        
        # Ensemble (weighted average - TCN slightly higher weight)
        if self.use_tcn and self.tcn_model is not None:
            ensemble_prob = (lgbm_up_prob * 0.45 + tcn_up_prob * 0.55)
        else:
            ensemble_prob = lgbm_up_prob
        
        # ==================== FLAT PREDICTION FILTER ====================
        if DynamicThreshold.should_skip_flat_prediction(ensemble_prob):
            self.trades_filtered += 1
            return self._hold_result("flat_prediction", vol_regime, 
                                    lgbm_up_prob, tcn_up_prob, ensemble_prob, threshold)
        
        # ==================== SIGNAL GENERATION ====================
        if ensemble_prob > threshold:
            signal = 'BUY'
            confidence = ensemble_prob
        elif ensemble_prob < (1 - threshold):
            signal = 'SELL'
            confidence = 1 - ensemble_prob
        else:
            self.trades_filtered += 1
            return self._hold_result("below_threshold", vol_regime,
                                    lgbm_up_prob, tcn_up_prob, ensemble_prob, threshold)
        
        # ==================== ADAPTIVE TP/SL ====================
        atr = vol_regime.get('atr_5', df_feat['atr_5'].iloc[-1])
        tp_sl = AdaptiveTPSL.calculate(atr, vol_regime, self.symbol)
        
        self.trades_passed += 1
        
        return {
            'signal': signal,
            'confidence': round(confidence, 4),
            'lgbm_prob': round(lgbm_up_prob, 4),
            'tcn_prob': round(tcn_up_prob, 4) if self.use_tcn else None,
            'ensemble_prob': round(ensemble_prob, 4),
            'threshold': threshold,
            'gated': True,
            'vol_regime': vol_regime['regime'],
            'atr_ratio': vol_regime['atr_ratio'],
            'tp_sl': tp_sl,
            'session_optimal': SessionFilter.is_optimal_session(self.symbol),
            'filter_stats': {
                'passed': self.trades_passed,
                'filtered': self.trades_filtered
            }
        }
    
    def _hold_result(self, reason: str, vol_regime: dict = None,
                    lgbm_prob: float = 0.5, tcn_prob: float = 0.5,
                    ensemble_prob: float = 0.5, threshold: float = 0.55) -> dict:
        """Generate HOLD result with metadata."""
        return {
            'signal': 'HOLD',
            'confidence': 0.5,
            'lgbm_prob': round(lgbm_prob, 4),
            'tcn_prob': round(tcn_prob, 4) if self.use_tcn else None,
            'ensemble_prob': round(ensemble_prob, 4),
            'threshold': threshold,
            'gated': False,
            'reason': reason,
            'vol_regime': vol_regime['regime'] if vol_regime else 'unknown',
            'filter_stats': {
                'passed': self.trades_passed,
                'filtered': self.trades_filtered
            }
        }
    
    def predict_proba(self, df: pd.DataFrame) -> tuple:
        """Get raw probabilities."""
        result = self.predict(df)
        up_prob = result['ensemble_prob']
        return (1 - up_prob, up_prob)
    
    def get_filter_stats(self) -> dict:
        """Get trade filtering statistics."""
        total = self.trades_passed + self.trades_filtered
        return {
            'total_signals': total,
            'passed': self.trades_passed,
            'filtered': self.trades_filtered,
            'filter_rate': round(self.trades_filtered / total * 100, 1) if total > 0 else 0
        }
    
    def reset_stats(self):
        """Reset filter statistics."""
        self.trades_filtered = 0
        self.trades_passed = 0
    
    def save(self, path: str):
        """Save model."""
        joblib.dump({
            'lgbm_model': self.lgbm_model,
            'tcn_weights': self.tcn_model.get_weights() if self.tcn_model else None,
            'scaler': self.scaler,
            'features': self.features,
            'prediction_horizon': self.prediction_horizon,
            'base_threshold': self.base_threshold,
            'tcn_timesteps': self.tcn_timesteps,
            'use_tcn': self.use_tcn,
            'symbol': self.symbol,
            'is_trained': self.is_trained
        }, path)
    
    def load(self, path: str):
        """Load model."""
        data = joblib.load(path)
        self.lgbm_model = data['lgbm_model']
        self.scaler = data['scaler']
        self.features = data['features']
        self.prediction_horizon = data['prediction_horizon']
        self.base_threshold = data['base_threshold']
        self.tcn_timesteps = data['tcn_timesteps']
        self.use_tcn = data['use_tcn']
        self.symbol = data.get('symbol', 'BTC')
        self.is_trained = data['is_trained']
        
        if data['tcn_weights'] is not None and self.use_tcn:
            self.tcn_model = build_optimized_tcn(
                input_shape=(self.tcn_timesteps, len(self.features)),
                num_filters=24, kernel_size=3, num_blocks=4, dropout=0.18
            )
            self.tcn_model.set_weights(data['tcn_weights'])


# ==================== STANDALONE TEST ====================
if __name__ == "__main__":
    print("=" * 70)
    print("BTC HFT PREDICTOR v3.0 - OPTIMIZED TCN + LightGBM")
    print("=" * 70)
    
    np.random.seed(42)
    n = 1500
    price = 40000 + np.cumsum(np.random.randn(n) * 50)
    
    df = pd.DataFrame({
        'open': price + np.random.randn(n) * 10,
        'high': price + np.abs(np.random.randn(n) * 30),
        'low': price - np.abs(np.random.randn(n) * 30),
        'close': price,
        'volume': np.random.randint(100, 10000, n).astype(float)
    })
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    print(f"\nData: {len(df)} candles")
    print(f"LightGBM: {HAS_LGBM} | TensorFlow: {HAS_TF}")
    
    # Test volatility regime
    vol = VolatilityRegime.classify(df)
    print(f"\nVolatility Regime: {vol['regime']} (ATR ratio: {vol['atr_ratio']})")
    
    # Test session filter
    session_mult = SessionFilter.get_session_multiplier("BTC")
    print(f"Session Multiplier: {session_mult}")
    
    # Test dynamic threshold
    threshold = DynamicThreshold.calculate(vol, "BTC")
    print(f"Dynamic Threshold: {threshold}")
    
    # Train predictor
    predictor = BTCHFTPredictor(
        prediction_horizon=2,
        base_threshold=0.55,
        use_tcn=HAS_TF,
        tcn_timesteps=30,
        symbol="BTC"
    )
    
    print("\nTraining optimized TCN + LightGBM...")
    metrics = predictor.train(df, retrain_cycles=3)
    
    print(f"\nResults:")
    print(f"  LightGBM: {metrics['lgbm_accuracy']:.2%}")
    if metrics['tcn_accuracy']:
        print(f"  TCN:      {metrics['tcn_accuracy']:.2%}")
    print(f"  Features: {metrics['features']}")
    
    # Test predictions
    print("\nRunning 100 predictions...")
    for i in range(100):
        _ = predictor.predict(df.iloc[:500 + i*10])
    
    stats = predictor.get_filter_stats()
    print(f"\nFilter Stats:")
    print(f"  Total:    {stats['total_signals']}")
    print(f"  Passed:   {stats['passed']}")
    print(f"  Filtered: {stats['filtered']} ({stats['filter_rate']}%)")
    
    # Final prediction
    pred = predictor.predict(df)
    print(f"\nFinal Prediction:")
    print(f"  Signal:    {pred['signal']}")
    print(f"  Conf:      {pred.get('confidence', 0.5):.2%}")
    print(f"  Threshold: {pred.get('threshold', 0.55)}")
    print(f"  Regime:    {pred.get('vol_regime', 'unknown')}")
    if pred.get('tp_sl'):
        print(f"  TP/SL:     {pred['tp_sl']}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
