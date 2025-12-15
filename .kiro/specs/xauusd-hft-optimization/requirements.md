# Requirements Document

## Introduction

This document specifies the requirements for optimizing the XAUUSD 1-minute high-frequency trading (HFT) model for maximum profitability. The system transforms raw OHLCV data into a compact, high-signal microstructure feature set designed for 1–3 candle prediction horizons. The objective is stable, repeatable profit generation at high volume (2000 trades/day) using a fast, compact LSTM/XGBoost hybrid that captures short-term imbalances.

## Glossary

- **XAUUSD_HFT_System**: The high-frequency trading system for gold (XAU/USD) on 1-minute timeframes
- **OHLCV**: Open, High, Low, Close, Volume - standard candlestick data
- **Microstructure Features**: Short-term price action patterns derived from candle anatomy and orderflow
- **LSTM**: Long Short-Term Memory neural network for sequence prediction
- **XGBoost**: Gradient boosting algorithm for feature ranking and probability prediction
- **ATR**: Average True Range - volatility measurement indicator
- **EMA**: Exponential Moving Average - trend indicator
- **Z-Score Normalization**: Statistical normalization method (value - mean) / standard_deviation
- **Ensemble Model**: Combined prediction from multiple ML models
- **Session Encoding**: Numeric representation of trading sessions (Asia=0, London=1, NY=2, Overlap=3)
- **Volatility Regime**: Market state classification based on ATR behavior

## Requirements

### Requirement 1: Micro-Momentum Feature Engineering

**User Story:** As a quantitative trader, I want micro-momentum features calculated from raw OHLCV data, so that I can capture short-term price direction signals.

#### Acceptance Criteria

1. WHEN the XAUUSD_HFT_System receives new OHLCV data, THE XAUUSD_HFT_System SHALL calculate 1-candle log return as ln(close[t] / close[t-1]).
2. WHEN the XAUUSD_HFT_System receives new OHLCV data, THE XAUUSD_HFT_System SHALL calculate 3-candle return as (close[t] - close[t-3]) / close[t-3].
3. WHEN the XAUUSD_HFT_System receives new OHLCV data, THE XAUUSD_HFT_System SHALL calculate 5-candle return as (close[t] - close[t-5]) / close[t-5].
4. WHEN the XAUUSD_HFT_System receives new OHLCV data, THE XAUUSD_HFT_System SHALL calculate candle body ratio as abs(close - open) / (high - low).
5. WHEN the XAUUSD_HFT_System receives new OHLCV data, THE XAUUSD_HFT_System SHALL calculate upper wick ratio as (high - max(open, close)) / (high - low).
6. WHEN the XAUUSD_HFT_System receives new OHLCV data, THE XAUUSD_HFT_System SHALL calculate lower wick ratio as (min(open, close) - low) / (high - low).

### Requirement 2: Microstructure Trend Features

**User Story:** As a quantitative trader, I want microstructure trend features based on fast EMAs, so that I can detect short-term trend direction and deviation.

#### Acceptance Criteria

1. WHEN the XAUUSD_HFT_System calculates trend features, THE XAUUSD_HFT_System SHALL compute EMA6 slope as (EMA6[t] - EMA6[t-1]) / EMA6[t-1].
2. WHEN the XAUUSD_HFT_System calculates trend features, THE XAUUSD_HFT_System SHALL compute price deviation from EMA6 as (close - EMA6) / EMA6 expressed as percentage.
3. WHEN the XAUUSD_HFT_System calculates trend features, THE XAUUSD_HFT_System SHALL compute price deviation from EMA21 as (close - EMA21) / EMA21 expressed as percentage.

### Requirement 3: Volatility and Expansion Features

**User Story:** As a quantitative trader, I want volatility and expansion features, so that I can identify favorable trading conditions and avoid low-volatility periods.

#### Acceptance Criteria

1. WHEN the XAUUSD_HFT_System calculates volatility features, THE XAUUSD_HFT_System SHALL compute ATR(5) normalized by dividing ATR(5) by the current close price.
2. WHEN the XAUUSD_HFT_System calculates volatility features, THE XAUUSD_HFT_System SHALL compute ATR expansion ratio as ATR(5) / ATR(20).
3. WHEN the XAUUSD_HFT_System calculates volatility features, THE XAUUSD_HFT_System SHALL compute Bollinger bandwidth with period 10 as (upper_band - lower_band) / middle_band.

### Requirement 4: Volume and Orderflow Features

**User Story:** As a quantitative trader, I want volume and orderflow features, so that I can detect buying/selling pressure and volume anomalies.

#### Acceptance Criteria

1. WHEN the XAUUSD_HFT_System calculates volume features, THE XAUUSD_HFT_System SHALL compute volume delta as current_volume minus previous_volume.
2. WHEN the XAUUSD_HFT_System calculates volume features, THE XAUUSD_HFT_System SHALL compute 3-candle volume change percentage as (volume[t] - volume[t-3]) / volume[t-3].
3. WHEN the XAUUSD_HFT_System calculates volume features, THE XAUUSD_HFT_System SHALL set volume spike flag to 1 when current volume exceeds 2 times the 20-period rolling mean volume, otherwise set to 0.

### Requirement 5: Market Regime Features

**User Story:** As a quantitative trader, I want market regime features, so that I can adapt trading behavior to different market conditions and sessions.

#### Acceptance Criteria

1. WHEN the XAUUSD_HFT_System calculates regime features, THE XAUUSD_HFT_System SHALL encode trading session as: Asia=0, London=1, NY=2, Overlap=3.
2. WHEN the XAUUSD_HFT_System calculates regime features, THE XAUUSD_HFT_System SHALL set volatility regime flag to 1 when ATR(5) exceeds ATR(5) from the previous candle, otherwise set to 0.

### Requirement 6: Feature Normalization

**User Story:** As a quantitative trader, I want all features normalized using z-score, so that the model receives stationary, comparable inputs.

#### Acceptance Criteria

1. WHEN the XAUUSD_HFT_System prepares features for model input, THE XAUUSD_HFT_System SHALL apply z-score normalization to all numeric features using a rolling window of 60 periods.
2. WHEN the XAUUSD_HFT_System prepares features for model input, THE XAUUSD_HFT_System SHALL convert all price values to returns to ensure stationarity.

### Requirement 7: Target Variable Definition

**User Story:** As a quantitative trader, I want a binary direction label for 1-3 candle prediction, so that the model predicts short-term price movement.

#### Acceptance Criteria

1. WHEN the XAUUSD_HFT_System creates training labels, THE XAUUSD_HFT_System SHALL set target to 1 when price closes higher within the next 1-3 candles compared to current close.
2. WHEN the XAUUSD_HFT_System creates training labels, THE XAUUSD_HFT_System SHALL set target to 0 when price closes lower within the next 1-3 candles compared to current close.

### Requirement 8: LSTM Model Architecture

**User Story:** As a quantitative trader, I want an LSTM model with specific architecture, so that I can capture temporal patterns in microstructure features.

#### Acceptance Criteria

1. WHEN the XAUUSD_HFT_System builds the LSTM model, THE XAUUSD_HFT_System SHALL create architecture with LSTM(64, return_sequences=True), Dropout(0.2), LSTM(32), Dropout(0.2), Dense(16, relu), Dense(1, sigmoid).
2. WHEN the XAUUSD_HFT_System trains the LSTM model, THE XAUUSD_HFT_System SHALL apply L2 regularization to LSTM layers.
3. WHEN the XAUUSD_HFT_System trains the LSTM model, THE XAUUSD_HFT_System SHALL use early stopping with patience of 3 epochs.
4. WHEN the XAUUSD_HFT_System starts a new training cycle, THE XAUUSD_HFT_System SHALL randomize weights without warm-start cycling.

### Requirement 9: XGBoost Model Configuration

**User Story:** As a quantitative trader, I want XGBoost configured for feature ranking and probability prediction, so that I can ensemble with LSTM output.

#### Acceptance Criteria

1. WHEN the XAUUSD_HFT_System trains XGBoost, THE XAUUSD_HFT_System SHALL tune hyperparameters: max_depth, learning_rate, n_estimators, subsample, colsample_bytree.
2. WHEN the XAUUSD_HFT_System uses XGBoost, THE XAUUSD_HFT_System SHALL output feature importance rankings.
3. WHEN the XAUUSD_HFT_System uses XGBoost, THE XAUUSD_HFT_System SHALL remove features with importance below 0.01 threshold.
4. WHEN the XAUUSD_HFT_System uses XGBoost, THE XAUUSD_HFT_System SHALL output probability predictions between 0 and 1.

### Requirement 10: Ensemble Prediction Logic

**User Story:** As a quantitative trader, I want LSTM and XGBoost predictions combined with weighted ensemble, so that I can maximize prediction accuracy.

#### Acceptance Criteria

1. WHEN the XAUUSD_HFT_System generates final prediction, THE XAUUSD_HFT_System SHALL calculate final_prediction as (0.6 × lstm_output) + (0.4 × xgb_output).
2. WHEN final_prediction exceeds 0.55, THE XAUUSD_HFT_System SHALL generate a trade signal.
3. WHEN final_prediction is 0.55 or below, THE XAUUSD_HFT_System SHALL skip the trade opportunity.

### Requirement 11: Training Pipeline

**User Story:** As a quantitative trader, I want a complete training pipeline, so that I can train and deploy models efficiently.

#### Acceptance Criteria

1. WHEN the XAUUSD_HFT_System executes training pipeline, THE XAUUSD_HFT_System SHALL normalize all features using z-score.
2. WHEN the XAUUSD_HFT_System executes training pipeline, THE XAUUSD_HFT_System SHALL create rolling 60-step sequences for LSTM input.
3. WHEN the XAUUSD_HFT_System executes training pipeline, THE XAUUSD_HFT_System SHALL train XGBoost on flattened feature vectors.
4. WHEN the XAUUSD_HFT_System executes training pipeline, THE XAUUSD_HFT_System SHALL calibrate decision threshold to maximize profit rather than accuracy.
5. WHEN the XAUUSD_HFT_System executes training pipeline, THE XAUUSD_HFT_System SHALL perform walk-forward backtesting with minimum 5 splits.
6. WHEN the XAUUSD_HFT_System completes training, THE XAUUSD_HFT_System SHALL export both LSTM and XGBoost models to disk.

### Requirement 12: Live Execution Rules

**User Story:** As a quantitative trader, I want specific execution rules for XAUUSD scalping, so that I can maximize profitability in live trading.

#### Acceptance Criteria

1. WHILE the XAUUSD_HFT_System holds a position, THE XAUUSD_HFT_System SHALL exit within 3-5 minutes maximum hold time.
2. IF volatility collapses below ATR threshold during a trade, THEN THE XAUUSD_HFT_System SHALL exit the position immediately.
3. WHILE trading during Asia session, THE XAUUSD_HFT_System SHALL reduce position size by 50%.
4. WHILE trading during London-NY overlap, THE XAUUSD_HFT_System SHALL increase position size by 100%.
5. WHEN the XAUUSD_HFT_System enters a trade, THE XAUUSD_HFT_System SHALL set take-profit between 0.07% and 0.12% of entry price.
6. WHEN the XAUUSD_HFT_System enters a trade, THE XAUUSD_HFT_System SHALL set stop-loss between 0.05% and 0.08% of entry price.

### Requirement 13: Performance Targets

**User Story:** As a quantitative trader, I want defined performance targets, so that I can measure model effectiveness.

#### Acceptance Criteria

1. THE XAUUSD_HFT_System SHALL achieve prediction accuracy between 60% and 67% on 1-3 candle horizons.
2. THE XAUUSD_HFT_System SHALL maintain inference speed capable of processing 2000 trades per day.
3. THE XAUUSD_HFT_System SHALL adapt to regime shifts within 10 candles of regime change detection.
