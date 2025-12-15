"""
================================================================================
STRATOSPHERE v3.0 - BTC 5-MINUTE MOMENTUM MODE
================================================================================
BTCUSDm ONLY - TCN + LightGBM ensemble
XAU/Gold uses separate xauusd_hft_bot.py (XGBoost)

EXACT SPECIFICATION (DO NOT CHANGE):
- Base candle feed: 1-minute
- Effective timeframe: 5-minute (aggregated)
- Prediction horizon: 3 candles = ~15 minutes ahead

This is the sweet spot where:
- BTC moves are large enough to beat $0.18 spread
- Noise is reduced
- TCN + LightGBM performs best
- Filters stop choking the system
- Trades actually appear
- Expectancy becomes positive
================================================================================
"""

from dataclasses import dataclass, field
from typing import Tuple, List
import os


@dataclass
class BTCConfig:
    """
    BTCUSDm — 5-Minute Momentum Mode (OPTIMIZED v3.1)
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │ CORE TIMEFRAME SPECIFICATION (DO NOT MODIFY)                        │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Base candle feed      : 1-minute (raw data)                         │
    │ Effective timeframe   : 5-minute (rolling aggregation)              │
    │ Prediction horizon    : 3 candles = ~15 minutes forward intent      │
    └─────────────────────────────────────────────────────────────────────┘
    
    OPTIMIZED FOR: 25-50 trades/day with positive expectancy
    Strategy: Adaptive regime-aware momentum trading
    
    Key Changes (v3.1):
    - Dynamic regime-dependent confidence thresholds
    - Volatility-scaled spread multiplier (1.8× in trending, 3× in compression)
    - Model agreement secondary entry condition
    - Asymmetric TP/SL optimization
    - Reduced flat-prediction band during volatility expansion
    """
    SYMBOL: str = "BTCUSDm"
    
    # ==================== TIMEFRAME (EXACT - DO NOT CHANGE) ====================
    BASE_TIMEFRAME: int = 1           # M1 base data feed (raw)
    EFFECTIVE_WINDOW: int = 5         # 5-minute rolling aggregation window
    PREDICTION_HORIZON: int = 3       # 3 candles ahead = ~15 minutes forward
    
    # ==================== SPREAD REQUIREMENTS (REGIME-ADAPTIVE) ====================
    TYPICAL_SPREAD_USD: float = 18.0  # ~$18 typical spread for BTCUSDm
    MAX_SPREAD_USD: float = 22.0      # Max acceptable spread (pause if exceeded)
    
    # Volatility-scaled spread multipliers (NOT hard 3×)
    # In trending/expanding regimes: allow trades when ATR-based move >= 1.8× spread
    # In compression/low vol: require >= 2.5× spread
    SPREAD_MULT_TRENDING: float = 1.8   # Trending/high vol regimes
    SPREAD_MULT_MEDIUM: float = 2.2     # Medium volatility
    SPREAD_MULT_LOW: float = 2.8        # Low volatility (stricter)
    MIN_EXPECTED_MOVE_USD: float = 32.0 # 1.8 * $18 = $32.4 minimum in trending
    
    # ==================== TRADE FREQUENCY TARGET (INCREASED) ====================
    TARGET_TRADES_MIN: int = 25       # Increased from 10
    TARGET_TRADES_MAX: int = 50       # Increased from 30
    
    # ==================== CONFIDENCE THRESHOLDS (REGIME-DEPENDENT) ====================
    # Dynamic thresholds based on volatility regime:
    # - Medium/High vol: allow trades at lower confidence (0.53-0.56)
    # - Low vol: keep stricter confidence (≥0.58)
    BASE_THRESHOLD: float = 0.55      # Base confidence (lowered from 0.58)
    
    # Regime-specific thresholds
    THRESHOLD_LOW_VOL: float = 0.58   # Stricter in low volatility
    THRESHOLD_MEDIUM_VOL: float = 0.55  # Standard in medium volatility
    THRESHOLD_HIGH_VOL: float = 0.53  # Relaxed in high volatility (momentum)
    THRESHOLD_EXTREME_VOL: float = 0.56  # Slightly stricter in extreme (protection)
    
    MIN_THRESHOLD: float = 0.53       # Floor (lowered from 0.56)
    MAX_THRESHOLD: float = 0.60       # Ceiling (extreme conditions)
    
    # Flat prediction margin (regime-adaptive)
    FLAT_MARGIN_DEFAULT: float = 0.06  # Default: |prob - 0.5| < 0.06
    FLAT_MARGIN_EXPANDING: float = 0.03  # Reduced to 0.03 when volatility expanding
    
    # ==================== MODEL AGREEMENT SECONDARY ENTRY ====================
    # Allow trades if both models agree on direction, even if combined prob is slightly below threshold
    ENABLE_MODEL_AGREEMENT_ENTRY: bool = True
    MODEL_AGREEMENT_THRESHOLD: float = 0.51  # Both models must be > 0.51 for same direction
    MODEL_AGREEMENT_COMBINED_MIN: float = 0.52  # Combined prob floor when using agreement
    
    # ==================== RISK MANAGEMENT (ASYMMETRIC TP/SL) ====================
    # Asymmetric optimization:
    # - Wider TP during volatility expansion (trend continuation)
    # - Tighter SL during entry to increase win-rate
    
    # Base TP/SL ranges
    TP_MIN_USD: float = 45.0          # Lowered from 60 (allows more trades)
    TP_MAX_USD: float = 180.0         # Increased from 150 (wider in expansion)
    SL_MIN_USD: float = 25.0          # Tighter SL (lowered from 40)
    SL_MAX_USD: float = 60.0          # Reduced from 80 (tighter risk)
    
    # Asymmetric multipliers by regime
    TP_MULT_EXPANSION: float = 1.4    # Wider TP in volatility expansion
    TP_MULT_CONTRACTION: float = 0.9  # Tighter TP in contraction
    SL_MULT_ENTRY: float = 0.8        # Tighter SL at entry (increase win-rate)
    
    MIN_RR_RATIO: float = 1.3         # Lowered from 1.5 (allows more trades)
    MIN_RR_RATIO_TRENDING: float = 1.5  # Higher R:R in trending markets
    
    # ==================== VOLATILITY REQUIREMENTS ====================
    MIN_ATR_USD: float = 15.0         # Lowered from 20 (allows more trades)
    VOLATILITY_EXPANSION_BIAS: bool = True
    
    # Volatility expansion detection
    VOL_EXPANSION_THRESHOLD: float = 1.15  # ATR5/ATR20 > 1.15 = expanding
    VOL_CONTRACTION_THRESHOLD: float = 0.85  # ATR5/ATR20 < 0.85 = contracting
    
    # ==================== SESSION PREFERENCES (UTC) ====================
    PREFERRED_SESSIONS: List[Tuple[int, int]] = field(default_factory=lambda: [(8, 20)])  # Extended: London+NY
    REDUCED_SESSIONS: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 5)])  # Asia - lower quality
    
    # ==================== ML SETTINGS ====================
    TCN_TIMESTEPS: int = 30
    RETRAIN_INTERVAL: int = 500
    MIN_TRAIN_SAMPLES: int = 50000
    
    # ==================== LOGGING ====================
    LOG_ALL_DECISIONS: bool = True
    LOG_FILTER_REJECTIONS: bool = True


# XAU/Gold uses separate XGBoost system - see xauusd_hft_bot.py
# This engine is BTC-ONLY with TCN+LightGBM


@dataclass
class StratosphereConfig:
    """
    Global Stratosphere Configuration
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │ BTCUSDm EXCLUSIVE - TCN+LightGBM Ensemble                           │
    │ Optimized for momentum & volatility expansion                       │
    │ XAU/Gold uses separate xauusd_hft_bot.py (XGBoost)                  │
    └─────────────────────────────────────────────────────────────────────┘
    
    GLOBAL PREDICTION HORIZON: 3 candles (~15 minutes)
    """
    # ==================== SYMBOL CONFIGURATION ====================
    SYMBOLS: Tuple[str] = ("BTCUSDm",)  # BTC ONLY
    MAGIC_NUMBER: int = 333333
    
    # ==================== GLOBAL PREDICTION HORIZON ====================
    # This MUST match BTCConfig.PREDICTION_HORIZON
    PREDICTION_HORIZON: int = 3       # 3 candles = ~15 minutes forward
    
    # ==================== MODEL ARCHITECTURE ====================
    # TCN captures momentum patterns, LightGBM captures regime features
    TCN_WEIGHT: float = 0.6           # TCN weight (momentum-focused)
    LGBM_WEIGHT: float = 0.4          # LightGBM weight (regime-focused)
    
    # ==================== TRAINING ====================
    WALK_FORWARD_SPLITS: int = 5
    EARLY_STOPPING_PATIENCE: int = 15
    CALIBRATION_METHOD: str = "isotonic"  # Probability calibration
    
    # ==================== EXECUTION ====================
    MAX_CONCURRENT_POSITIONS: int = 1  # Single position for BTC
    TICK_SLEEP_MS: float = 1.0
    
    # ==================== MONITORING & LOGGING ====================
    # NO SILENT SKIPS - Full decision logging
    LOG_ALL_REJECTIONS: bool = True
    LOG_ALL_DECISIONS: bool = True
    LOG_FILTER_DETAILS: bool = True
    
    # ==================== AUTO-PAUSE CONDITIONS ====================
    AUTO_PAUSE_ON_SPREAD_SPIKE: bool = True
    AUTO_PAUSE_ON_VOL_COLLAPSE: bool = True
    AUTO_PAUSE_ON_CONFIDENCE_DEGRADE: bool = True
    
    # ==================== PATHS ====================
    MODEL_DIR: str = "stratosphere_v3_models"
    LOG_FILE: str = "stratosphere_v3.log"
    TRADE_LOG: str = "stratosphere_v3_trades.csv"
    DECISION_LOG: str = "stratosphere_v3_decisions.csv"
    
    # ==================== FEATURE ENGINEERING ====================
    CORRELATION_THRESHOLD: float = 0.85
    
    def __post_init__(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        # Validate prediction horizon consistency
        assert self.PREDICTION_HORIZON == 3, "PREDICTION_HORIZON must be 3 (15 min forward)"


# Singleton instances - BTC ONLY
BTC_CFG = BTCConfig()
GLOBAL_CFG = StratosphereConfig()
