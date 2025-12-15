# STRATOSPHERE v3.0 - BTCUSDm 5-Minute Momentum Engine

## Overview

STRATOSPHERE v3.0 is a **BTCUSDm-EXCLUSIVE** trading engine optimized for **momentum and volatility expansion** trades. It uses TCN + LightGBM ensemble architecture with spread-aware execution filters and comprehensive risk management.

**For Gold/XAU trading, use `xauusd_hft_bot.py` (XGBoost-based system).**

## Core Specification

```
┌─────────────────────────────────────────────────────────────────────┐
│ CORE SPECIFICATION (DO NOT MODIFY)                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Base candle feed      : 1-minute (raw data)                         │
│ Effective timeframe   : 5-minute (rolling aggregation)              │
│ Prediction horizon    : 3 candles = ~15 minutes forward intent      │
│ Spread requirement    : Expected move >= 3× spread (~$0.54)         │
│ Confidence threshold  : 0.56 - 0.60 (raised for momentum)           │
│ Target trades/day     : 10-30 (quality over quantity)               │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Symbol | BTCUSDm | BTC micro account |
| Base Timeframe | 1-minute | Raw data feed |
| Effective Window | 5-minute | Rolling aggregation |
| Prediction Horizon | 3 candles | ~15 minutes forward |
| Typical Spread | $0.18 | Execution cost |
| Min Expected Move | $0.54 | 3× spread requirement |
| Base Threshold | 0.58 | Confidence requirement |
| Threshold Range | 0.56-0.60 | Dynamic adjustment |
| Flat Margin | 0.08 | Skip indecisive signals |
| Target Trades | 10-30/day | Quality over quantity |

## Strategy: Momentum & Volatility Expansion

This engine is optimized for **momentum and volatility expansion** trades, NOT micro-scalping:

- **Momentum**: Captures directional moves when price is trending
- **Volatility Expansion**: Trades when ATR is expanding (ATR5 > ATR20)
- **Spread-Aware**: No trade unless expected move exceeds 3× spread
- **Higher Confidence**: Raised thresholds (0.56-0.60) for quality signals

## Model Architecture

```
final_prob = 0.6 × TCN + 0.4 × LightGBM
```

- **TCN (60%)**: Captures local price structure & volatility bursts
- **LightGBM (40%)**: Captures non-linear feature interactions & regimes
- **Probability Calibration**: Isotonic regression for reliable confidence scores
- **Prediction Horizon**: 3 candles (~15 minutes) - enforced globally

## Feature Engineering

### Core Features
- Log returns (1, 3, 5 periods)
- Rolling volatility (ATR5, ATR10, ATR20)
- Wick-to-body ratios
- Candle range expansion
- EMA deviation (6, 12 periods)
- Volume delta & acceleration
- Volatility regime flags (low/medium/high/extreme)
- Session encoding (Asia/London/NY)

### BTC Momentum Features
- 5-minute aggregated returns
- 5-minute high/low range
- Volatility expansion ratio (ATR5/ATR20)
- Momentum (3-period, 5-period)
- Momentum strength (normalized by ATR)
- Volatility expansion flag
- ATR acceleration

All features are z-score normalized with correlation pruning (>85% → drop).

## Execution Filters (Spread-Aware)

Applied in order:
1. **Spread Filter**: Reject if expected move < 3× spread ($0.54)
2. **Volatility Regime Filter**: Skip low-volatility compression
3. **Session Filter**: Prefer London/NY overlap
4. **Flat Prediction Filter**: Skip if |prob − 0.5| < 0.08
5. **Confidence Gate**: Dynamic thresholds (0.56-0.60)

**NO SILENT SKIPS** - Every decision is logged with full details.

## Risk Management

| Parameter | Value |
|-----------|-------|
| TP Range | $0.60 - $1.50 |
| SL Range | $0.40 - $0.80 |
| Min R:R | 1.5:1 |
| Risk/Trade | 2% |
| Max Exposure | 50% |

**Never scalp inside spread** - All trades must have positive expected value.

## Safety & Monitoring

Auto-pause conditions:
- Spread spikes (>3× typical)
- Volatility collapse (low regime + ATR ratio < 0.5)
- Model confidence degradation

Full decision logging:
- Every signal evaluated
- Every rejection with reason
- Filter statistics
- Trade outcomes

## Usage

### Live Trading

```bash
# Using batch file
run_btc.bat

# Or directly
python -m stratosphere_v3.engine --symbol BTCUSDm
```

### Backtesting

```bash
python -m stratosphere_v3.backtest
```

### System Validation

```bash
python -m stratosphere_v3.test_system
```

## File Structure

```
stratosphere_v3/
├── __init__.py          # Package exports
├── config.py            # BTCUSDm configuration (PREDICTION_HORIZON=3)
├── features.py          # Momentum & expansion features
├── models.py            # TCN + LightGBM ensemble (horizon=3)
├── filters.py           # Spread-aware execution filters
├── risk.py              # Spread-aware risk management
├── engine.py            # Main trading engine
├── backtest.py          # Walk-forward backtester
├── monitor.py           # Monitoring & safety
└── test_system.py       # System validation
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Win Rate | 55-60% |
| Profit Factor | >1.5 |
| Max Drawdown | <15% |
| Sharpe Ratio | >1.5 |
| Trades/Day | 10-30 |
| Expectancy | Positive |

## Objective

**Fewer, higher-expectancy BTC trades** prioritizing:
- Economic viability (beat spread)
- Consistent profitability
- Quality over quantity
- Full transparency (no silent skips)

## Gold/XAU

For Gold trading, use the separate XGBoost-based system:
- File: `xauusd_hft_bot.py`
- Target: 200-600 trades/day
- Model: LSTM + XGBoost ensemble

## License

Proprietary - All rights reserved.
