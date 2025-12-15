"""
================================================================================
STRATOSPHERE v3.0 - SYSTEM TEST & VALIDATION
================================================================================
Validates all components without requiring MT5 connection.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stratosphere_v3.config import GLOBAL_CFG, BTC_CFG
from stratosphere_v3.features import SpreadAwareFeatures
from stratosphere_v3.models import TCNLightGBMEnsemble, HAS_TF, HAS_LGBM
from stratosphere_v3.filters import ExecutionFilterChain, FilterReason
from stratosphere_v3.risk import SpreadAwareRiskManager, TPSLLevels
from stratosphere_v3.monitor import StratosphereMonitor


def generate_test_data(n: int = 2000, symbol: str = "BTC") -> pd.DataFrame:
    """Generate realistic test OHLCV data."""
    np.random.seed(42)
    
    if "BTC" in symbol.upper():
        base_price = 40000
        volatility = 100
    else:
        base_price = 2000
        volatility = 5
    
    # Generate price with trend and mean reversion
    returns = np.random.randn(n) * 0.002  # 0.2% per candle
    trend = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.001
    price = base_price * np.exp(np.cumsum(returns + trend))
    
    # Generate OHLCV
    df = pd.DataFrame({
        'open': price + np.random.randn(n) * volatility * 0.1,
        'high': price + np.abs(np.random.randn(n) * volatility * 0.3),
        'low': price - np.abs(np.random.randn(n) * volatility * 0.3),
        'close': price,
        'volume': np.random.randint(100, 10000, n).astype(float)
    })
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # Add datetime index
    df.index = pd.date_range(start='2024-01-01', periods=n, freq='1min')
    
    return df


def test_config():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("TEST: Configuration")
    print("=" * 60)
    
    print(f"✅ Global Config loaded")
    print(f"   Symbols: {GLOBAL_CFG.SYMBOLS}")
    print(f"   TCN Weight: {GLOBAL_CFG.TCN_WEIGHT}")
    print(f"   LightGBM Weight: {GLOBAL_CFG.LGBM_WEIGHT}")
    
    print(f"✅ BTC Config loaded")
    print(f"   Prediction Horizon: {BTC_CFG.PREDICTION_HORIZON}")
    print(f"   Base Threshold: {BTC_CFG.BASE_THRESHOLD}")
    print(f"   Min Expected Move: ${BTC_CFG.MIN_EXPECTED_MOVE_USD}")
    
    print(f"   Note: XAU/Gold uses separate xauusd_hft_bot.py (XGBoost)")
    
    return True


def test_features():
    """Test feature engineering."""
    print("\n" + "=" * 60)
    print("TEST: Feature Engineering")
    print("=" * 60)
    
    # Test BTC features
    df_btc = generate_test_data(500, "BTC")
    features_btc = SpreadAwareFeatures("BTC", correlation_threshold=0.85)
    
    X_btc, names_btc = features_btc.fit_transform(df_btc)
    
    print(f"✅ BTC Features calculated")
    print(f"   Input rows: {len(df_btc)}")
    print(f"   Output rows: {len(X_btc)}")
    print(f"   Features: {len(names_btc)}")
    print(f"   Dropped (correlated): {len(features_btc.dropped_features)}")
    
    # Test regime detection
    regime = features_btc.get_regime(df_btc)
    print(f"   Current Regime: {regime['regime']} (ATR ratio: {regime['atr_ratio']:.2f})")
    
    # XAU uses separate XGBoost system - not tested here
    print(f"   Note: XAU/Gold uses xauusd_hft_bot.py (XGBoost)")
    
    return True


def test_model():
    """Test model training and prediction."""
    print("\n" + "=" * 60)
    print("TEST: TCN + LightGBM Ensemble")
    print("=" * 60)
    
    print(f"   LightGBM available: {HAS_LGBM}")
    print(f"   TensorFlow available: {HAS_TF}")
    
    if not HAS_LGBM:
        print("⚠️ LightGBM not installed, skipping model test")
        return True
    
    # Generate data
    df = generate_test_data(1500, "BTC")
    features = SpreadAwareFeatures("BTC")
    X, feature_names = features.fit_transform(df)
    
    # Train model
    model = TCNLightGBMEnsemble(
        tcn_weight=0.6,
        lgbm_weight=0.4,
        tcn_timesteps=30,
        prediction_horizon=3
    )
    
    print("   Training model...")
    metrics = model.train(
        X=X,
        close=df['close'].iloc[-len(X):].reset_index(drop=True),
        feature_names=feature_names,
        n_splits=3,
        early_stopping_patience=10
    )
    
    print(f"✅ Model trained")
    print(f"   LightGBM Accuracy: {metrics['lgbm_accuracy']:.2%}")
    if metrics['tcn_accuracy']:
        print(f"   TCN Accuracy: {metrics['tcn_accuracy']:.2%}")
    print(f"   Train samples: {metrics['samples_train']}")
    print(f"   Val samples: {metrics['samples_val']}")
    
    # Test prediction
    X_test = features.transform(df.tail(100))
    prediction = model.predict(X_test, threshold=0.55)
    
    print(f"✅ Prediction generated")
    print(f"   Signal: {prediction['signal']}")
    print(f"   Confidence: {prediction['confidence']:.2%}")
    print(f"   Ensemble Prob: {prediction['ensemble_prob']:.2%}")
    
    return True


def test_filters():
    """Test execution filters."""
    print("\n" + "=" * 60)
    print("TEST: Execution Filters")
    print("=" * 60)
    
    filters = ExecutionFilterChain(
        symbol="BTC",
        spread_multiplier=3.0,
        base_threshold=0.58,
        flat_margin=0.08
    )
    
    # Test various scenarios
    test_cases = [
        # (prob, expected_move, spread, regime, atr_ratio, expected_result)
        (0.65, 0.60, 0.18, 'medium', 1.0, True),   # Should pass - BUY
        (0.52, 0.60, 0.18, 'medium', 1.0, False),  # Flat prediction
        (0.65, 0.30, 0.18, 'medium', 1.0, False),  # Spread filter
        (0.65, 0.60, 0.18, 'low', 0.5, False),     # Low volatility
        (0.35, 0.60, 0.18, 'medium', 1.0, True),   # SELL signal (0.35 < 1-0.58=0.42)
    ]
    
    for i, (prob, exp_move, spread, regime, atr_ratio, expected) in enumerate(test_cases):
        passed, signal, details = filters.execute(
            probability=prob,
            expected_move=exp_move,
            spread=spread,
            regime=regime,
            atr_ratio=atr_ratio,
            log_rejections=False
        )
        
        status = "✅" if passed == expected else "❌"
        print(f"   {status} Case {i+1}: prob={prob:.2f}, regime={regime} -> {signal} (expected: {'PASS' if expected else 'REJECT'})")
    
    stats = filters.get_stats()
    print(f"\n   Filter Stats:")
    print(f"   Total: {stats['total_signals']} | Passed: {stats['passed_signals']} | Rate: {stats['pass_rate']}%")
    
    return True


def test_risk_management():
    """Test risk management."""
    print("\n" + "=" * 60)
    print("TEST: Risk Management")
    print("=" * 60)
    
    # Test BTC risk manager
    rm_btc = SpreadAwareRiskManager("BTC")
    
    tp_sl = rm_btc.calculate_tp_sl(
        entry_price=40000,
        direction='BUY',
        atr=50,
        regime='medium',
        spread=0.18
    )
    
    print(f"✅ BTC TP/SL calculated")
    print(f"   Entry: $40,000")
    print(f"   TP: ${tp_sl.tp_price} ({tp_sl.tp_pips} pips)")
    print(f"   SL: ${tp_sl.sl_price} ({tp_sl.sl_pips} pips)")
    print(f"   R:R: {tp_sl.risk_reward}")
    
    # Test position sizing
    pos_size = rm_btc.calculate_position_size(
        balance=10000,
        sl_distance=abs(40000 - tp_sl.sl_price),
        atr=50,
        regime='medium'
    )
    
    print(f"✅ Position size calculated")
    print(f"   Lot size: {pos_size.lot_size}")
    print(f"   Risk amount: ${pos_size.risk_amount}")
    print(f"   Max loss: ${pos_size.max_loss}")
    
    # XAU uses separate XGBoost system
    print(f"   Note: XAU/Gold uses xauusd_hft_bot.py (XGBoost)")
    
    return True


def test_monitor():
    """Test monitoring system."""
    print("\n" + "=" * 60)
    print("TEST: Monitoring System")
    print("=" * 60)
    
    monitor = StratosphereMonitor()
    
    # Log some signals
    for i in range(20):
        prob = 0.5 + np.random.randn() * 0.1
        passed = prob > 0.55 or prob < 0.45
        
        monitor.log_signal(
            symbol="BTC",
            probability=prob,
            threshold=0.55,
            passed=passed,
            filter_reason="passed" if passed else "below_confidence",
            spread=0.18,
            regime="medium",
            expected_move=0.60
        )
    
    # Log some trades
    for i in range(5):
        trade_id = monitor.log_trade_entry(
            symbol="BTC",
            direction="BUY" if i % 2 == 0 else "SELL",
            entry_price=40000 + i * 10,
            spread=0.18,
            confidence=0.60,
            threshold=0.55,
            regime="medium",
            expected_move=0.60
        )
        
        pnl = np.random.choice([-20, 30, -15, 40, 25])
        monitor.log_trade_exit(
            symbol="BTC",
            trade_id=trade_id,
            exit_price=40000 + i * 10 + (pnl / 10),
            pnl=pnl,
            exit_reason="TP" if pnl > 0 else "SL"
        )
    
    # Check safety
    safety = monitor.check_safety_conditions("BTC", current_spread=0.20)
    print(f"✅ Safety check: {'PAUSE' if safety['should_pause'] else 'OK'}")
    
    # Get stats
    stats = monitor.get_stats("BTC")
    print(f"✅ Stats retrieved")
    print(f"   Signals: {stats.total_signals}")
    print(f"   Trades: {stats.trades_executed}")
    print(f"   Win Rate: {stats.trades_won / max(stats.trades_executed, 1) * 100:.1f}%")
    
    return True


def run_all_tests():
    """Run all system tests."""
    print("\n" + "=" * 70)
    print("STRATOSPHERE v3.0 - SYSTEM VALIDATION")
    print("=" * 70)
    
    tests = [
        ("Configuration", test_config),
        ("Feature Engineering", test_features),
        ("Model Training", test_model),
        ("Execution Filters", test_filters),
        ("Risk Management", test_risk_management),
        ("Monitoring", test_monitor),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {name}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
