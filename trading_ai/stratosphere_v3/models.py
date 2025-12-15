"""
================================================================================
STRATOSPHERE v3.0 - BTC 5-MINUTE MOMENTUM MODE
================================================================================
TCN + LightGBM ENSEMBLE - BTC ONLY

Base candle feed      : 1-minute
Effective timeframe   : 5-minute (aggregated)
Prediction horizon    : 3 candles = ~15 minutes ahead

Primary Architecture:
- Temporal Convolutional Network (TCN): Captures local price structure & volatility bursts
- LightGBM: Captures non-linear feature interactions & regimes

Ensemble: final_prob = 0.6 * TCN + 0.4 * LightGBM

EXACT NUMBERS:
- PREDICTION_HORIZON = 3 (in config)
- horizon = 3 (in ML engine)
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

# LightGBM
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: LightGBM not installed")

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
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("WARNING: TensorFlow not installed")


class TCNBlock(keras.layers.Layer):
    """
    Optimized TCN block with:
    - Causal dilated convolutions
    - Spatial dropout for regularization
    - L2 regularization
    - Residual connections
    """
    
    def __init__(self, filters: int, kernel_size: int, dilation_rate: int, 
                 dropout_rate: float = 0.2, **kwargs):
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
        
        # Residual projection if needed
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


def build_tcn(input_shape: Tuple[int, int], 
              num_filters: int = 24, 
              kernel_size: int = 3, 
              num_blocks: int = 4, 
              dropout: float = 0.18) -> Model:
    """
    Build optimized TCN for price prediction.
    
    Architecture:
    - 4 TCN blocks with increasing dilation (1, 2, 4, 8)
    - Global average pooling
    - Dense layers with dropout
    - Softmax output (2 classes: down, up)
    """
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Stack TCN blocks with exponential dilation
    for i in range(num_blocks):
        dilation_rate = 2 ** i
        x = TCNBlock(num_filters, kernel_size, dilation_rate, dropout)(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(dropout)(x)
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0008),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


class TCNLightGBMEnsemble:
    """
    Ensemble model combining TCN and LightGBM.
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │ PREDICTION HORIZON: 3 candles = ~15 minutes forward                 │
    │ Optimized for momentum & volatility expansion (NOT micro-scalping)  │
    └─────────────────────────────────────────────────────────────────────┘
    
    Features:
    - Walk-forward validation
    - Class imbalance handling
    - Probability calibration (Platt/Isotonic)
    - Early stopping
    - No repeated warm-start loops
    """
    
    # GLOBAL CONSTANT - DO NOT CHANGE
    REQUIRED_PREDICTION_HORIZON: int = 3  # 3 candles = ~15 minutes
    
    def __init__(self, 
                 tcn_weight: float = 0.6,
                 lgbm_weight: float = 0.4,
                 tcn_timesteps: int = 30,
                 prediction_horizon: int = 3,
                 calibration_method: str = "isotonic"):
        
        # Enforce prediction horizon = 3
        if prediction_horizon != self.REQUIRED_PREDICTION_HORIZON:
            import logging
            log = logging.getLogger("STRATOSPHERE")
            log.warning(f"Overriding prediction_horizon {prediction_horizon} -> {self.REQUIRED_PREDICTION_HORIZON}")
            prediction_horizon = self.REQUIRED_PREDICTION_HORIZON
        
        self.tcn_weight = tcn_weight
        self.lgbm_weight = lgbm_weight
        self.tcn_timesteps = tcn_timesteps
        self.prediction_horizon = prediction_horizon  # Always 3
        self.calibration_method = calibration_method
        
        self.tcn_model: Optional[Model] = None
        self.lgbm_model = None
        self.calibrator: Optional[IsotonicRegression] = None
        
        self.feature_names: List[str] = []
        self.is_trained = False
        
        # Training metrics
        self.metrics: Dict = {}
    
    def _prepare_targets(self, close: pd.Series) -> np.ndarray:
        """Create binary targets: 1 if price goes up in horizon, 0 otherwise."""
        future_close = close.shift(-self.prediction_horizon)
        targets = (future_close > close).astype(int)
        return targets.values
    
    def _prepare_tcn_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for TCN input."""
        X_seq, y_seq = [], []
        for i in range(self.tcn_timesteps, len(X)):
            X_seq.append(X[i - self.tcn_timesteps:i])
            y_seq.append([1 - y[i], y[i]])  # One-hot: [down_prob, up_prob]
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X: np.ndarray, close: pd.Series, feature_names: List[str],
              n_splits: int = 5, early_stopping_patience: int = 15) -> Dict:
        """
        Train ensemble with walk-forward validation.
        
        Args:
            X: Feature matrix (already scaled)
            close: Close prices for target calculation
            feature_names: List of feature names
            n_splits: Number of walk-forward splits
            early_stopping_patience: Early stopping patience
        
        Returns:
            Training metrics dict
        """
        self.feature_names = feature_names
        
        # Prepare targets
        y = self._prepare_targets(close)
        
        # Remove NaN targets
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask].astype(int)
        close_valid = close.iloc[valid_mask].reset_index(drop=True)
        
        if len(X) < 500:
            raise ValueError(f"Insufficient data: {len(X)} samples")
        
        # Walk-forward split (use last split for final training)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(X))
        train_idx, val_idx = splits[-1]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Class weights for imbalance
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # ==================== TRAIN LIGHTGBM ====================
        lgbm_acc = 0.0
        if HAS_LGBM:
            self.lgbm_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                min_child_weight=10,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight='balanced',
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            
            self.lgbm_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(early_stopping_patience, verbose=False)]
            )
            
            lgbm_acc = (self.lgbm_model.predict(X_val) == y_val).mean()
        
        # ==================== TRAIN TCN ====================
        tcn_acc = 0.0
        if HAS_TF:
            X_tcn_train, y_tcn_train = self._prepare_tcn_sequences(X_train, y_train)
            X_tcn_val, y_tcn_val = self._prepare_tcn_sequences(X_val, y_val)
            
            if len(X_tcn_train) > 100:
                keras.backend.clear_session()
                
                self.tcn_model = build_tcn(
                    input_shape=(self.tcn_timesteps, len(feature_names)),
                    num_filters=24,
                    kernel_size=3,
                    num_blocks=4,
                    dropout=0.18
                )
                
                callbacks = [
                    EarlyStopping(
                        monitor='val_accuracy',
                        patience=early_stopping_patience,
                        restore_best_weights=True
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=0.0001
                    )
                ]
                
                tcn_class_weights = {0: class_weight_dict[0], 1: class_weight_dict[1]}
                
                self.tcn_model.fit(
                    X_tcn_train, y_tcn_train,
                    validation_data=(X_tcn_val, y_tcn_val),
                    epochs=50,
                    batch_size=32,
                    class_weight=tcn_class_weights,
                    callbacks=callbacks,
                    verbose=0
                )
                
                tcn_acc = self.tcn_model.evaluate(X_tcn_val, y_tcn_val, verbose=0)[1]
        
        # ==================== PROBABILITY CALIBRATION ====================
        if self.calibration_method == "isotonic":
            # Get ensemble probabilities on validation set
            val_probs = self._get_ensemble_proba(X_val)
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(val_probs, y_val)
        
        self.is_trained = True
        
        self.metrics = {
            'lgbm_accuracy': round(lgbm_acc, 4),
            'tcn_accuracy': round(tcn_acc, 4) if HAS_TF else None,
            'samples_train': len(y_train),
            'samples_val': len(y_val),
            'features': len(feature_names),
            'prediction_horizon': self.prediction_horizon,
            'class_balance': {
                'train_up_pct': round(y_train.mean() * 100, 1),
                'val_up_pct': round(y_val.mean() * 100, 1)
            }
        }
        
        return self.metrics
    
    def _get_ensemble_proba(self, X: np.ndarray) -> np.ndarray:
        """Get raw ensemble probability (before calibration)."""
        probs = np.full(len(X), 0.5)
        
        # LightGBM probability
        lgbm_prob = np.full(len(X), 0.5)
        if self.lgbm_model is not None:
            lgbm_prob = self.lgbm_model.predict_proba(X)[:, 1]
        
        # TCN probability (need sequences)
        tcn_prob = np.full(len(X), 0.5)
        if self.tcn_model is not None and len(X) >= self.tcn_timesteps:
            # Only predict for samples where we have full sequence
            for i in range(self.tcn_timesteps, len(X)):
                seq = X[i - self.tcn_timesteps:i].reshape(1, self.tcn_timesteps, -1)
                tcn_prob[i] = self.tcn_model.predict(seq, verbose=0)[0, 1]
        
        # Ensemble
        probs = self.tcn_weight * tcn_prob + self.lgbm_weight * lgbm_prob
        
        return probs
    
    def predict_proba(self, X: np.ndarray) -> float:
        """
        Get calibrated probability for single prediction.
        
        Args:
            X: Feature matrix (last tcn_timesteps rows used for TCN)
        
        Returns:
            Calibrated probability of price going up
        """
        if not self.is_trained:
            return 0.5
        
        # LightGBM prediction (use last row)
        lgbm_prob = 0.5
        if self.lgbm_model is not None:
            lgbm_prob = self.lgbm_model.predict_proba(X[-1:])[:, 1][0]
        
        # TCN prediction (use last sequence)
        tcn_prob = 0.5
        if self.tcn_model is not None and len(X) >= self.tcn_timesteps:
            seq = X[-self.tcn_timesteps:].reshape(1, self.tcn_timesteps, -1)
            tcn_prob = self.tcn_model.predict(seq, verbose=0)[0, 1]
        
        # Ensemble
        raw_prob = self.tcn_weight * tcn_prob + self.lgbm_weight * lgbm_prob
        
        # Calibrate
        if self.calibrator is not None:
            calibrated = self.calibrator.predict([raw_prob])[0]
            return float(calibrated)
        
        return float(raw_prob)
    
    def predict(self, X: np.ndarray, threshold: float = 0.55) -> Dict:
        """
        Get prediction with signal and confidence.
        
        Returns:
            Dict with signal, confidence, probabilities
        """
        prob = self.predict_proba(X)
        
        # Get individual model probabilities for logging
        lgbm_prob = 0.5
        if self.lgbm_model is not None:
            lgbm_prob = self.lgbm_model.predict_proba(X[-1:])[:, 1][0]
        
        tcn_prob = 0.5
        if self.tcn_model is not None and len(X) >= self.tcn_timesteps:
            seq = X[-self.tcn_timesteps:].reshape(1, self.tcn_timesteps, -1)
            tcn_prob = self.tcn_model.predict(seq, verbose=0)[0, 1]
        
        # Determine signal
        if prob > threshold:
            signal = 'BUY'
            confidence = prob
        elif prob < (1 - threshold):
            signal = 'SELL'
            confidence = 1 - prob
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        return {
            'signal': signal,
            'confidence': round(confidence, 4),
            'ensemble_prob': round(prob, 4),
            'lgbm_prob': round(lgbm_prob, 4),
            'tcn_prob': round(tcn_prob, 4),
            'threshold': threshold
        }
    
    def save(self, path: str):
        """Save model to file."""
        data = {
            'lgbm_model': self.lgbm_model,
            'tcn_weights': self.tcn_model.get_weights() if self.tcn_model else None,
            'calibrator': self.calibrator,
            'feature_names': self.feature_names,
            'tcn_timesteps': self.tcn_timesteps,
            'prediction_horizon': self.prediction_horizon,
            'tcn_weight': self.tcn_weight,
            'lgbm_weight': self.lgbm_weight,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }
        joblib.dump(data, path)
    
    def load(self, path: str):
        """Load model from file."""
        data = joblib.load(path)
        
        self.lgbm_model = data['lgbm_model']
        self.calibrator = data['calibrator']
        self.feature_names = data['feature_names']
        self.tcn_timesteps = data['tcn_timesteps']
        self.prediction_horizon = data['prediction_horizon']
        self.tcn_weight = data['tcn_weight']
        self.lgbm_weight = data['lgbm_weight']
        self.metrics = data['metrics']
        self.is_trained = data['is_trained']
        
        # Rebuild TCN and load weights
        if data['tcn_weights'] is not None and HAS_TF:
            self.tcn_model = build_tcn(
                input_shape=(self.tcn_timesteps, len(self.feature_names)),
                num_filters=24, kernel_size=3, num_blocks=4, dropout=0.18
            )
            self.tcn_model.set_weights(data['tcn_weights'])
