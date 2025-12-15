"""
================================================================================
STRATOSPHERE v3.1 - BTCUSDm 5-MINUTE MOMENTUM ENGINE (OPTIMIZED)
================================================================================
BTCUSDm EXCLUSIVE - TCN + LightGBM Ensemble

┌─────────────────────────────────────────────────────────────────────┐
│ CORE SPECIFICATION                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Base candle feed      : 1-minute (raw data)                         │
│ Effective timeframe   : 5-minute (rolling aggregation)              │
│ Prediction horizon    : 3 candles = ~15 minutes forward intent      │
│ Spread requirement    : Regime-adaptive (1.8× trending, 2.8× low)   │
│ Confidence threshold  : Regime-dependent (0.53-0.58)                │
│ Target trades/day     : 25-50 (increased from 10-30)                │
└─────────────────────────────────────────────────────────────────────┘

v3.1 Optimizations:
- Dynamic regime-dependent confidence thresholds
- Volatility-scaled spread multiplier (1.8× trending vs 3× fixed)
- Model agreement secondary entry condition
- Asymmetric TP/SL (wider TP in expansion, tighter SL at entry)
- Reduced flat-prediction band during volatility expansion
- Selective trade gating (not global rejection)

Strategy: Adaptive regime-aware momentum trading
Objective: Increased trade density with preserved positive expectancy

For Gold/XAU: Use xauusd_hft_bot.py (XGBoost-based)
================================================================================
"""

import os
import sys
import time
import threading
import queue
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Local imports
from .config import StratosphereConfig, BTCConfig, GLOBAL_CFG, BTC_CFG
from .features import SpreadAwareFeatures, aggregate_to_5min
from .models import TCNLightGBMEnsemble
from .filters import ExecutionFilterChain, FilterReason
from .risk import SpreadAwareRiskManager, VirtualOrderManager, TPSLLevels

# MT5 Import
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(GLOBAL_CFG.LOG_FILE, encoding='utf-8')
    ]
)
log = logging.getLogger("STRATOSPHERE")


@dataclass
class AssetState:
    """Per-asset trading state."""
    symbol: str
    config: object  # BTCConfig or XAUConfig
    features: SpreadAwareFeatures
    model: TCNLightGBMEnsemble
    filters: ExecutionFilterChain
    risk_manager: SpreadAwareRiskManager
    voms: VirtualOrderManager
    
    # State
    is_trained: bool = False
    candles_processed: int = 0
    last_candle_time: Optional[datetime] = None
    last_regime: str = "medium"
    last_spread: float = 0.0
    
    # Safety flags
    paused: bool = False
    pause_reason: str = ""
    
    # Stats
    trades_today: int = 0
    pnl_today: float = 0.0
    wins_today: int = 0
    losses_today: int = 0


class MT5Handler:
    """MetaTrader5 connection handler."""
    
    def __init__(self):
        self.connected = False
        self.symbol_info: Dict[str, any] = {}
        self.active_symbols: List[str] = []
    
    def connect(self) -> bool:
        if not HAS_MT5:
            log.error("MetaTrader5 not installed")
            return False
        
        for attempt in range(3):
            if mt5.initialize():
                self.connected = True
                info = mt5.account_info()
                log.info(f"MT5 Connected | Account: {info.login} | Balance: ${info.balance:.2f}")
                return True
            time.sleep(1)
        
        log.error("MT5 connection failed")
        return False
    
    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
    
    def enable_symbol(self, symbol: str) -> bool:
        """Enable symbol in Market Watch."""
        if mt5.symbol_select(symbol, True):
            sym_info = mt5.symbol_info(symbol)
            if sym_info:
                self.symbol_info[symbol] = sym_info
                self.active_symbols.append(symbol)
                log.info(f"✅ {symbol} | Spread: {sym_info.spread} | Min Lot: {sym_info.volume_min}")
                return True
        log.warning(f"⚠️ Could not enable {symbol}")
        return False
    
    def get_tick(self, symbol: str) -> Tuple[float, float, float]:
        """Get current bid, ask, spread."""
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return tick.bid, tick.ask, tick.ask - tick.bid
        return 0, 0, 0
    
    def get_balance(self) -> float:
        info = mt5.account_info()
        return info.balance if info else 0
    
    def fetch_candles(self, symbol: str, count: int = 200) -> pd.DataFrame:
        """Fetch M1 OHLCV data."""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def open_position(self, symbol: str, direction: str, lot: float) -> Optional[int]:
        """Open position with NO SL/TP (VOMS handles risk)."""
        bid, ask, _ = self.get_tick(symbol)
        
        if direction == "BUY":
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
            "sl": 0.0,
            "tp": 0.0,
            "deviation": 20,
            "magic": GLOBAL_CFG.MAGIC_NUMBER,
            "comment": "STRATOSPHERE_V3",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"🟢 OPEN | {symbol} | {direction} {lot} @ {price:.2f} | Ticket: {result.order}")
            return result.order
        
        error = mt5.last_error() if result is None else result.retcode
        log.error(f"Order failed: {error}")
        return None
    
    def close_position(self, ticket: int, symbol: str, reason: str = "") -> bool:
        """Close position by ticket."""
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
            "magic": GLOBAL_CFG.MAGIC_NUMBER,
            "comment": f"CLOSE_{reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"🔴 CLOSE | {symbol} | Ticket {ticket} | Reason: {reason} | P&L: ${pos.profit:.2f}")
            return True
        
        return False


class StratosphereEngine:
    """
    Main STRATOSPHERE v3.0 Trading Engine.
    
    Features:
    - Multi-asset support (BTC + XAU)
    - Continuous structure scanning
    - Spread-aware profitability optimization
    - Comprehensive monitoring and safety
    """
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or list(GLOBAL_CFG.SYMBOLS)
        self.mt5 = MT5Handler()
        self.assets: Dict[str, AssetState] = {}
        self.running = False
        
        # Global stats
        self.total_trades = 0
        self.total_pnl = 0.0
        
        # Threading
        self.inference_queue = queue.Queue(maxsize=100)
    
    def initialize(self) -> bool:
        """Initialize engine and all assets."""
        log.info("=" * 70)
        log.info("🚀 STRATOSPHERE v3.0 - SPREAD-AWARE PROFITABILITY ENGINE")
        log.info("=" * 70)
        log.info(f"Symbols: {', '.join(self.symbols)}")
        log.info(f"TCN Weight: {GLOBAL_CFG.TCN_WEIGHT} | LightGBM Weight: {GLOBAL_CFG.LGBM_WEIGHT}")
        log.info("=" * 70)
        
        # Connect to MT5
        if not self.mt5.connect():
            return False
        
        # Initialize each asset
        for symbol in self.symbols:
            if not self._initialize_asset(symbol):
                log.warning(f"⚠️ Failed to initialize {symbol}")
                continue
        
        if not self.assets:
            log.error("No assets initialized!")
            return False
        
        log.info(f"✅ Initialization complete | {len(self.assets)} assets active")
        return True
    
    def _initialize_asset(self, symbol: str) -> bool:
        """Initialize single asset - BTC ONLY."""
        log.info(f"📊 Initializing {symbol}...")
        
        # BTC ONLY - reject non-BTC symbols
        if "BTC" not in symbol.upper():
            log.warning(f"⚠️ {symbol} rejected - this engine is BTC ONLY. Use xauusd_hft_bot.py for Gold.")
            return False
        
        # Enable symbol
        if not self.mt5.enable_symbol(symbol):
            return False
        
        # Get config - BTC only
        config = BTC_CFG
        
        # Fetch initial 1-minute data (will be aggregated to 5-min for training)
        min_samples = config.MIN_TRAIN_SAMPLES
        df_1m = self.mt5.fetch_candles(symbol, min(min_samples, 10000))
        
        if len(df_1m) < 500:
            log.warning(f"⚠️ {symbol}: Insufficient data ({len(df_1m)} M1 candles)")
            return False
        
        log.info(f"   Fetched {len(df_1m)} M1 candles for 5-min aggregation")
        
        # Initialize components
        features = SpreadAwareFeatures(symbol, GLOBAL_CFG.CORRELATION_THRESHOLD)
        
        # Ensure prediction horizon is exactly 3 (global + asset config)
        prediction_horizon = config.PREDICTION_HORIZON
        assert prediction_horizon == 3, f"PREDICTION_HORIZON must be 3, got {prediction_horizon}"
        assert GLOBAL_CFG.PREDICTION_HORIZON == 3, f"Global PREDICTION_HORIZON must be 3"
        
        model = TCNLightGBMEnsemble(
            tcn_weight=GLOBAL_CFG.TCN_WEIGHT,
            lgbm_weight=GLOBAL_CFG.LGBM_WEIGHT,
            tcn_timesteps=config.TCN_TIMESTEPS,
            prediction_horizon=prediction_horizon,  # Always 3
            calibration_method=GLOBAL_CFG.CALIBRATION_METHOD
        )
        
        filters = ExecutionFilterChain(
            symbol=symbol,
            # Regime-adaptive spread multipliers (v3.1)
            spread_mult_trending=config.SPREAD_MULT_TRENDING,  # 1.8×
            spread_mult_medium=config.SPREAD_MULT_MEDIUM,      # 2.2×
            spread_mult_low=config.SPREAD_MULT_LOW,            # 2.8×
            # Regime-dependent confidence thresholds
            threshold_low_vol=config.THRESHOLD_LOW_VOL,        # 0.58
            threshold_medium_vol=config.THRESHOLD_MEDIUM_VOL,  # 0.55
            threshold_high_vol=config.THRESHOLD_HIGH_VOL,      # 0.53
            threshold_extreme_vol=config.THRESHOLD_EXTREME_VOL,  # 0.56
            # Adaptive flat prediction margin
            flat_margin_default=config.FLAT_MARGIN_DEFAULT,    # 0.06
            flat_margin_expanding=config.FLAT_MARGIN_EXPANDING,  # 0.03
            # Model agreement secondary entry
            enable_model_agreement=config.ENABLE_MODEL_AGREEMENT_ENTRY,
            model_agreement_threshold=config.MODEL_AGREEMENT_THRESHOLD,
            model_agreement_combined_min=config.MODEL_AGREEMENT_COMBINED_MIN
        )
        
        risk_manager = SpreadAwareRiskManager(symbol)
        voms = VirtualOrderManager(risk_manager)
        
        # Create asset state
        asset = AssetState(
            symbol=symbol,
            config=config,
            features=features,
            model=model,
            filters=filters,
            risk_manager=risk_manager,
            voms=voms
        )
        
        # Train model with 1-minute data (aggregated to 5-min internally)
        try:
            self._train_asset(asset, df_1m)
            asset.is_trained = True
        except Exception as e:
            log.error(f"Training failed for {symbol}: {e}")
            return False
        
        self.assets[symbol] = asset
        log.info(f"✅ {symbol} ready")
        return True
    
    def _train_asset(self, asset: AssetState, df_1m: pd.DataFrame):
        """Train model for asset using 5-minute aggregated data."""
        log.info(f"🔄 Training {asset.symbol}...")
        log.info(f"   Base: 1-min | Effective: 5-min | Horizon: 3 candles (~15 min)")
        log.info(f"   Optimized for: Momentum & Volatility Expansion")
        
        # Aggregate 1-minute to 5-minute (EFFECTIVE TIMEFRAME)
        df = aggregate_to_5min(df_1m)
        log.info(f"   Aggregated {len(df_1m)} M1 candles -> {len(df)} M5 candles")
        
        # Calculate features on 5-minute data
        X, feature_names = asset.features.fit_transform(df)
        
        # Train model
        metrics = asset.model.train(
            X=X,
            close=df['close'].iloc[-len(X):].reset_index(drop=True),
            feature_names=feature_names,
            n_splits=GLOBAL_CFG.WALK_FORWARD_SPLITS,
            early_stopping_patience=GLOBAL_CFG.EARLY_STOPPING_PATIENCE
        )
        
        log.info(f"✅ {asset.symbol} | LightGBM: {metrics['lgbm_accuracy']:.2%} | "
                f"TCN: {metrics['tcn_accuracy']:.2%} | Features: {metrics['features']}")
        
        # Save model
        model_path = os.path.join(GLOBAL_CFG.MODEL_DIR, f"{asset.symbol}_model.pkl")
        asset.model.save(model_path)
        log.info(f"💾 Model saved: {model_path}")
    
    def run(self):
        """Main execution loop."""
        if not self.initialize():
            return
        
        self.running = True
        tick_count = 0
        
        log.info("🟢 ENGINE RUNNING - Waiting for signals...")
        
        try:
            while self.running:
                try:
                    tick_start = time.perf_counter()
                    
                    # Process each asset
                    for symbol, asset in self.assets.items():
                        self._process_asset(asset)
                    
                    # Status logging
                    tick_count += 1
                    if tick_count % 60000 == 0:
                        self._log_status()
                    
                    # Sleep
                    elapsed = time.perf_counter() - tick_start
                    sleep_time = max(0, GLOBAL_CFG.TICK_SLEEP_MS / 1000 - elapsed)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    log.error(f"Loop error: {e}")
                    time.sleep(1)
        
        except KeyboardInterrupt:
            log.info("Stopped by user")
        
        finally:
            self.shutdown()
    
    def _process_asset(self, asset: AssetState):
        """Process single asset - check exits and entries."""
        # Check VOMS exits
        exits = asset.voms.check_exits(self.mt5.get_tick)
        for ticket, exit_type, details in exits:
            if self.mt5.close_position(ticket, asset.symbol, exit_type.value):
                asset.voms.remove_position(ticket)
                
                # Update stats
                pnl = details.get('pnl_pips', 0) * asset.risk_manager.usd_per_pip_per_lot * 0.01
                asset.pnl_today += pnl
                asset.trades_today += 1
                if pnl > 0:
                    asset.wins_today += 1
                else:
                    asset.losses_today += 1
        
        # Skip if paused
        if asset.paused:
            return
        
        # Fetch 1-minute candles (base data feed)
        df_1m = self.mt5.fetch_candles(asset.symbol, 500)  # Need more for 5-min aggregation
        if len(df_1m) < 50:
            return
        
        # === AGGREGATE TO 5-MINUTE (EFFECTIVE TIMEFRAME) ===
        # Base: 1-minute | Effective: 5-minute | Horizon: 3 candles = ~15 min
        df = aggregate_to_5min(df_1m)
        if len(df) < 30:
            return
        
        current_candle = df.index[-1]
        
        # Only process on new 5-minute candle
        if asset.last_candle_time == current_candle:
            return
        
        asset.last_candle_time = current_candle
        asset.candles_processed += 1
        
        # === CONTINUOUS STRUCTURE SCANNING ===
        # Re-compute features, regimes, thresholds every 5-min candle
        
        # Get current market state
        bid, ask, spread = self.mt5.get_tick(asset.symbol)
        asset.last_spread = spread
        
        # Get regime
        regime_info = asset.features.get_regime(df)
        asset.last_regime = regime_info['regime']
        
        # Safety checks
        if self._check_safety_conditions(asset, spread, regime_info):
            return
        
        # Check for retrain
        if asset.candles_processed >= asset.config.RETRAIN_INTERVAL:
            self._retrain_asset(asset)
        
        # Skip if position already open
        if asset.voms.get_position_count(asset.symbol) > 0:
            return
        
        # === GENERATE SIGNAL (NO SILENT SKIPS) ===
        try:
            X = asset.features.transform(df)
            prediction = asset.model.predict(X)
            
            # Get expected move for spread filter (regime-adaptive)
            atr = regime_info.get('atr_5', 0)
            atr_ratio = regime_info.get('atr_ratio', 1.0)
            expected_move = asset.risk_manager.get_expected_move(atr, regime_info['regime'], atr_ratio)
            
            # Get regime-adaptive spread multiplier for logging
            spread_mult = asset.risk_manager.get_spread_multiplier(regime_info['regime'], atr_ratio)
            min_required_move = spread * spread_mult
            
            # Execute filter chain with model agreement support (v3.1)
            passed, signal, filter_details = asset.filters.execute(
                probability=prediction['ensemble_prob'],
                expected_move=expected_move,
                spread=spread,
                regime=regime_info['regime'],
                atr_ratio=atr_ratio,
                tcn_prob=prediction.get('tcn_prob'),    # For model agreement
                lgbm_prob=prediction.get('lgbm_prob'),  # For model agreement
                log_rejections=True  # Always log - no silent skips
            )
            
            # Log every decision (not just rejections)
            if GLOBAL_CFG.LOG_ALL_DECISIONS:
                decision = "TRADE" if passed else "SKIP"
                reason = filter_details.get('rejection_reason', 'passed')
                entry_type = filter_details.get('entry_type', 'none')
                log.debug(f"DECISION | {asset.symbol} | {decision} | "
                         f"Prob: {prediction['ensemble_prob']:.1%} | "
                         f"Expected: ${expected_move:.2f} vs Min: ${min_required_move:.2f} (×{spread_mult:.1f}) | "
                         f"Entry: {entry_type} | Reason: {reason}")
            
            if passed and signal in ['BUY', 'SELL']:
                self._execute_entry(asset, signal, prediction, regime_info, spread)
            
            # Status logging every 30 candles (~2.5 hours)
            if asset.candles_processed % 30 == 0:
                stats = asset.filters.get_stats()
                is_expanding = regime_info.get('is_expanding', False)
                vol_state = "EXPANDING" if is_expanding else ("CONTRACTING" if regime_info.get('is_contracting', False) else "STABLE")
                log.info(f"📊 {asset.symbol} | Candles: {asset.candles_processed} | "
                        f"Regime: {regime_info['regime']} ({vol_state}) | "
                        f"ATR Ratio: {atr_ratio:.2f} | "
                        f"Prob: {prediction['ensemble_prob']:.1%} | "
                        f"Pass Rate: {stats['pass_rate']}% | "
                        f"Agreement Entries: {stats.get('passed_via_agreement', 0)} | "
                        f"Trades Today: {asset.trades_today}")
        
        except Exception as e:
            log.error(f"{asset.symbol} prediction error: {e}")
    
    def _execute_entry(self, asset: AssetState, signal: str, prediction: Dict, 
                       regime_info: Dict, spread: float):
        """Execute trade entry with asymmetric TP/SL (v3.1)."""
        balance = self.mt5.get_balance()
        bid, ask, _ = self.mt5.get_tick(asset.symbol)
        
        entry_price = ask if signal == 'BUY' else bid
        atr = regime_info.get('atr_5', 0)
        atr_ratio = regime_info.get('atr_ratio', 1.0)
        
        # Calculate asymmetric TP/SL (v3.1)
        tp_sl = asset.risk_manager.calculate_tp_sl(
            entry_price=entry_price,
            direction=signal,
            atr=atr,
            regime=regime_info['regime'],
            spread=spread,
            atr_ratio=atr_ratio  # For asymmetric adjustment
        )
        
        # Calculate position size
        pos_size = asset.risk_manager.calculate_position_size(
            balance=balance,
            sl_distance=abs(entry_price - tp_sl.sl_price),
            atr=atr,
            regime=regime_info['regime']
        )
        
        # Open position
        ticket = self.mt5.open_position(asset.symbol, signal, pos_size.lot_size)
        
        if ticket:
            # Register with VOMS
            asset.voms.add_position(
                ticket=ticket,
                symbol=asset.symbol,
                direction=signal,
                entry_price=entry_price,
                lot_size=pos_size.lot_size,
                tp_sl=tp_sl
            )
            
            log.info(f"📈 {asset.symbol} | {signal} {pos_size.lot_size} | "
                    f"Conf: {prediction['confidence']:.1%} | "
                    f"TP: {tp_sl.tp_price} | SL: {tp_sl.sl_price} | "
                    f"R:R: {tp_sl.risk_reward}")
    
    def _check_safety_conditions(self, asset: AssetState, spread: float, 
                                  regime_info: Dict) -> bool:
        """Check safety conditions and pause if needed."""
        # Get spread thresholds from config
        typical_spread = getattr(asset.config, 'TYPICAL_SPREAD_USD', 18.0)
        max_spread = getattr(asset.config, 'MAX_SPREAD_USD', 20.0)
        
        # Spread spike detection - pause only if spread exceeds MAX_SPREAD_USD
        if GLOBAL_CFG.AUTO_PAUSE_ON_SPREAD_SPIKE:
            if spread > max_spread:
                asset.paused = True
                asset.pause_reason = f"Spread spike: ${spread:.2f} > max ${max_spread:.2f}"
                log.warning(f"⚠️ {asset.symbol} PAUSED: {asset.pause_reason}")
                return True
        
        # Volatility collapse
        if GLOBAL_CFG.AUTO_PAUSE_ON_VOL_COLLAPSE:
            if regime_info['regime'] == 'low' and regime_info['atr_ratio'] < 0.5:
                asset.paused = True
                asset.pause_reason = f"Volatility collapse: ATR ratio {regime_info['atr_ratio']:.2f}"
                log.warning(f"⚠️ {asset.symbol} PAUSED: {asset.pause_reason}")
                return True
        
        # Auto-unpause if conditions normalize
        if asset.paused:
            if spread <= max_spread and regime_info['regime'] != 'low':
                asset.paused = False
                asset.pause_reason = ""
                log.info(f"✅ {asset.symbol} RESUMED - Spread: ${spread:.2f}")
        
        return asset.paused
    
    def _retrain_asset(self, asset: AssetState):
        """Retrain asset model with 5-minute aggregated data."""
        log.info(f"🔄 Retraining {asset.symbol} (5-min momentum mode)...")
        
        # Fetch 1-minute data (will be aggregated to 5-min in _train_asset)
        df_1m = self.mt5.fetch_candles(asset.symbol, min(asset.config.MIN_TRAIN_SAMPLES, 10000))
        if len(df_1m) < 500:
            return
        
        try:
            self._train_asset(asset, df_1m)
            asset.candles_processed = 0
        except Exception as e:
            log.error(f"Retrain failed for {asset.symbol}: {e}")
    
    def _log_status(self):
        """Log engine status."""
        log.info("=" * 50)
        for symbol, asset in self.assets.items():
            win_rate = asset.wins_today / max(asset.trades_today, 1) * 100
            log.info(f"{symbol} | Trades: {asset.trades_today} | "
                    f"Win: {win_rate:.1f}% | P&L: ${asset.pnl_today:.2f} | "
                    f"Regime: {asset.last_regime}")
        log.info("=" * 50)
    
    def shutdown(self):
        """Shutdown engine."""
        self.running = False
        
        log.info("=" * 50)
        log.info("FINAL SUMMARY")
        for symbol, asset in self.assets.items():
            win_rate = asset.wins_today / max(asset.trades_today, 1) * 100
            filter_stats = asset.filters.get_stats()
            log.info(f"{symbol} | Trades: {asset.trades_today} | "
                    f"Win: {win_rate:.1f}% | P&L: ${asset.pnl_today:.2f} | "
                    f"Filter Pass Rate: {filter_stats['pass_rate']}%")
        log.info("=" * 50)
        
        self.mt5.disconnect()
        log.info("🔴 Engine shutdown complete")


# ==================== ENTRY POINT ====================
def main():
    """Main entry point - BTCUSDm ONLY."""
    import argparse
    
    parser = argparse.ArgumentParser(description='STRATOSPHERE v3.1 - BTCUSDm Momentum Engine (Optimized)')
    parser.add_argument('--symbol', default='BTCUSDm', help='BTC symbol to trade')
    
    args = parser.parse_args()
    
    # BTCUSDm ONLY
    symbols = [args.symbol]
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║     STRATOSPHERE v3.1 - BTCUSDm 5-MINUTE MOMENTUM ENGINE             ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║     Base: 1-min | Effective: 5-min | Horizon: 3 candles (~15 min)    ║
    ║     Spread: Regime-adaptive (1.8× trending, 2.8× compression)        ║
    ║     Confidence: Regime-dependent (0.53-0.58) | Trades: 25-50/day     ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║     v3.1 Optimizations:                                              ║
    ║     • Dynamic regime-dependent confidence thresholds                 ║
    ║     • Volatility-scaled spread multiplier (vs fixed 3×)              ║
    ║     • Model agreement secondary entry condition                      ║
    ║     • Asymmetric TP/SL (wider TP expansion, tighter SL entry)        ║
    ║     • Reduced flat-prediction band during volatility expansion       ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║     TCN + LightGBM Ensemble | Adaptive Filters | Full Logging        ║
    ║     For Gold/XAU: Use xauusd_hft_bot.py (XGBoost)                    ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    engine = StratosphereEngine(symbols=symbols)
    engine.run()


if __name__ == "__main__":
    main()
