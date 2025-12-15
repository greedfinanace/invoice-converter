"""
================================================================================
STRATOSPHERE v3.0 - BTCUSDm 5-MINUTE MOMENTUM ENGINE
================================================================================
Spread-Aware Profitability Engine - Optimized for Momentum & Volatility Expansion

┌─────────────────────────────────────────────────────────────────────┐
│ CORE SPECIFICATION                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Base candle feed      : 1-minute (raw data)                         │
│ Effective timeframe   : 5-minute (rolling aggregation)              │
│ Prediction horizon    : 3 candles = ~15 minutes forward             │
│ Spread requirement    : Expected move >= 3× spread (~$0.54)         │
│ Confidence threshold  : 0.56 - 0.60 (raised for momentum)           │
│ Target trades/day     : 10-30 (quality over quantity)               │
└─────────────────────────────────────────────────────────────────────┘

For Gold/XAU: Use xauusd_hft_bot.py (XGBoost-based)
"""

from .config import StratosphereConfig, BTCConfig, GLOBAL_CFG, BTC_CFG
from .features import SpreadAwareFeatures, aggregate_to_5min
from .models import TCNLightGBMEnsemble
from .filters import ExecutionFilterChain, FilterReason
from .risk import SpreadAwareRiskManager, VirtualOrderManager
from .engine import StratosphereEngine

__version__ = "3.0.0"
__all__ = [
    # Config
    "StratosphereConfig",
    "BTCConfig",
    "GLOBAL_CFG",
    "BTC_CFG",
    # Features
    "SpreadAwareFeatures",
    "aggregate_to_5min",
    # Models
    "TCNLightGBMEnsemble",
    # Filters
    "ExecutionFilterChain",
    "FilterReason",
    # Risk
    "SpreadAwareRiskManager",
    "VirtualOrderManager",
    # Engine
    "StratosphereEngine"
]
