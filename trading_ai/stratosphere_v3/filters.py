"""
================================================================================
STRATOSPHERE v3.1 - ADAPTIVE EXECUTION FILTERS (OPTIMIZED FOR TRADE DENSITY)
================================================================================
BTCUSDm 5-Minute Momentum Mode

┌─────────────────────────────────────────────────────────────────────┐
│ REGIME-ADAPTIVE TRADE VALIDATION (v3.1)                             │
│ Volatility-scaled spread multiplier: 1.8× trending, 2.8× compression│
│ Dynamic confidence: 0.53-0.58 based on regime                       │
│ Model agreement secondary entry condition                           │
└─────────────────────────────────────────────────────────────────────┘

Filter order:
1. Spread filter - Regime-adaptive (1.8× in trending, 2.8× in low vol)
2. Volatility regime filter - Selective gating (not global rejection)
3. Session filter - BTC: prefer London/NY overlap
4. Flat-prediction filter - Reduced band (0.03) during vol expansion
5. Confidence gate - Regime-dependent (0.53-0.58)
6. Model agreement - Secondary entry if both models agree

Key Changes (v3.1):
- Relaxed spread multiplier in trending regimes (1.8× vs 3×)
- Dynamic confidence thresholds by volatility regime
- Reduced flat-prediction band during expansion
- Model agreement allows entry below primary threshold
- Target: 25-50 trades/day with positive expectancy

NO SILENT SKIPS - Every rejection logs a reason with full details.
"""

import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

log = logging.getLogger("STRATOSPHERE")


class FilterReason(Enum):
    """Enumeration of all possible filter rejection reasons."""
    PASSED = "passed"
    PASSED_MODEL_AGREEMENT = "passed_model_agreement"  # Secondary entry via model agreement
    SPREAD_TOO_WIDE = "spread_exceeds_expected_move"
    LOW_VOLATILITY = "low_volatility_regime"
    EXTREME_VOLATILITY = "extreme_volatility_protection"
    UNFAVORABLE_SESSION = "unfavorable_session"
    BELOW_CONFIDENCE = "below_confidence_threshold"
    FLAT_PREDICTION = "flat_prediction_margin"
    CONSOLIDATION = "consolidation_detected"
    SPREAD_SPIKE = "spread_spike_detected"
    MODEL_DEGRADED = "model_confidence_degraded"
    MODEL_DISAGREEMENT = "models_disagree_on_direction"


@dataclass
class FilterResult:
    """Result of filter chain execution."""
    passed: bool
    reason: FilterReason
    details: Dict
    
    def __str__(self):
        if self.passed:
            return f"PASSED | {self.details}"
        return f"REJECTED: {self.reason.value} | {self.details}"


class SpreadFilter:
    """
    Filter 1: Regime-adaptive spread-aware trade gating.
    
    v3.1 Changes:
    - Volatility-scaled multiplier instead of hard 3×
    - Trending/high vol: 1.8× spread (allows more trades)
    - Medium vol: 2.2× spread
    - Low vol/compression: 2.8× spread (stricter)
    """
    
    def __init__(self, 
                 mult_trending: float = 1.8,
                 mult_medium: float = 2.2,
                 mult_low: float = 2.8):
        self.mult_trending = mult_trending
        self.mult_medium = mult_medium
        self.mult_low = mult_low
    
    def get_regime_multiplier(self, regime: str, atr_ratio: float) -> float:
        """Get spread multiplier based on regime and ATR ratio."""
        # Volatility expansion (atr_ratio > 1.15) = trending
        if atr_ratio > 1.15 or regime in ['high', 'extreme']:
            return self.mult_trending  # 1.8×
        elif regime == 'low' or atr_ratio < 0.85:
            return self.mult_low  # 2.8×
        else:
            return self.mult_medium  # 2.2×
    
    def check(self, expected_move: float, spread: float, 
              regime: str = 'medium', atr_ratio: float = 1.0) -> FilterResult:
        """
        Check if expected move exceeds regime-adaptive spread threshold.
        
        Args:
            expected_move: Expected price move in USD
            spread: Current spread in USD
            regime: Volatility regime ('low', 'medium', 'high', 'extreme')
            atr_ratio: ATR5/ATR20 ratio (>1.0 = expansion)
        """
        multiplier = self.get_regime_multiplier(regime, atr_ratio)
        min_required = spread * multiplier
        passed = expected_move >= min_required
        
        return FilterResult(
            passed=passed,
            reason=FilterReason.PASSED if passed else FilterReason.SPREAD_TOO_WIDE,
            details={
                'expected_move': round(expected_move, 4),
                'spread': round(spread, 4),
                'min_required': round(min_required, 4),
                'multiplier': round(multiplier, 2),
                'regime': regime,
                'atr_ratio': round(atr_ratio, 3)
            }
        )


class VolatilityRegimeFilter:
    """
    Filter 2: Selective volatility regime gating.
    
    v3.1 Changes:
    - NOT global rejection of low volatility
    - Only skip low vol when ATR ratio < 0.7 (severe compression)
    - Allow extreme volatility with stricter thresholds (handled by confidence gate)
    - Prioritize trade opportunity during valid momentum phases
    """
    
    def __init__(self, skip_severe_compression: bool = True, 
                 compression_threshold: float = 0.7):
        self.skip_severe_compression = skip_severe_compression
        self.compression_threshold = compression_threshold
    
    def check(self, regime: str, atr_ratio: float) -> FilterResult:
        """
        Check volatility regime with selective gating.
        
        Args:
            regime: 'low', 'medium', 'high', 'extreme'
            atr_ratio: ATR5/ATR10 ratio
        """
        # Only skip severe compression (atr_ratio < 0.7)
        # Regular low volatility is handled by stricter thresholds, not rejection
        if self.skip_severe_compression and regime == 'low' and atr_ratio < self.compression_threshold:
            return FilterResult(
                passed=False,
                reason=FilterReason.LOW_VOLATILITY,
                details={
                    'regime': regime, 
                    'atr_ratio': round(atr_ratio, 3),
                    'compression_threshold': self.compression_threshold,
                    'note': 'severe_compression_only'
                }
            )
        
        # Allow extreme volatility - confidence gate will apply stricter thresholds
        # This increases trade opportunity during momentum phases
        
        return FilterResult(
            passed=True,
            reason=FilterReason.PASSED,
            details={
                'regime': regime, 
                'atr_ratio': round(atr_ratio, 3),
                'is_expanding': atr_ratio > 1.15,
                'is_contracting': atr_ratio < 0.85
            }
        )


class SessionFilter:
    """
    Filter 3: Session-based trade gating.
    BTC: prefer London/NY
    XAU: reduce Asia exposure
    """
    
    # Session definitions (UTC hours)
    ASIA = (0, 8)
    LONDON = (8, 13)
    NY = (13, 21)
    
    def __init__(self, symbol: str = "BTC"):
        self.symbol = symbol.upper()
        self.is_btc = "BTC" in self.symbol
    
    def get_current_session(self) -> str:
        """Get current trading session."""
        hour = datetime.now(timezone.utc).hour
        
        if self.ASIA[0] <= hour < self.ASIA[1]:
            return 'asia'
        elif self.LONDON[0] <= hour < self.LONDON[1]:
            return 'london'
        elif self.NY[0] <= hour < self.NY[1]:
            return 'ny'
        return 'other'
    
    def get_threshold_multiplier(self) -> float:
        """
        Get threshold multiplier based on session.
        >1.0 = stricter, <1.0 = looser
        """
        session = self.get_current_session()
        weekday = datetime.now(timezone.utc).weekday()
        
        if self.is_btc:
            # BTC: prefer London/NY
            if weekday >= 5:  # Weekend
                return 1.08
            if session in ['london', 'ny']:
                return 0.97
            if session == 'asia':
                return 1.10
            return 1.0
        else:
            # XAU: reduce Asia exposure
            if session in ['london', 'ny']:
                return 0.95
            if session == 'asia':
                return 1.12
            return 1.0
    
    def check(self, strict_mode: bool = False) -> FilterResult:
        """
        Check if current session is favorable.
        
        Args:
            strict_mode: If True, reject unfavorable sessions entirely
        """
        session = self.get_current_session()
        multiplier = self.get_threshold_multiplier()
        
        if strict_mode:
            if self.is_btc and session == 'asia':
                return FilterResult(
                    passed=False,
                    reason=FilterReason.UNFAVORABLE_SESSION,
                    details={'session': session, 'multiplier': multiplier}
                )
            if not self.is_btc and session == 'asia':
                return FilterResult(
                    passed=False,
                    reason=FilterReason.UNFAVORABLE_SESSION,
                    details={'session': session, 'multiplier': multiplier}
                )
        
        return FilterResult(
            passed=True,
            reason=FilterReason.PASSED,
            details={'session': session, 'multiplier': multiplier}
        )


class ConfidenceGate:
    """
    Filter 5: Regime-dependent dynamic confidence threshold.
    
    v3.1 Thresholds (optimized for trade density):
    - Low vol: 0.58 (stricter - preserve capital)
    - Medium vol: 0.55 (standard)
    - High vol: 0.53 (relaxed - capture momentum)
    - Extreme vol: 0.56 (slightly stricter - protection)
    
    Target: 25-50 trades/day with positive expectancy
    """
    
    def __init__(self, 
                 threshold_low_vol: float = 0.58,
                 threshold_medium_vol: float = 0.55,
                 threshold_high_vol: float = 0.53,
                 threshold_extreme_vol: float = 0.56,
                 min_threshold: float = 0.53, 
                 max_threshold: float = 0.60):
        self.threshold_low_vol = threshold_low_vol
        self.threshold_medium_vol = threshold_medium_vol
        self.threshold_high_vol = threshold_high_vol
        self.threshold_extreme_vol = threshold_extreme_vol
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    def calculate_threshold(self, regime: str, atr_ratio: float, 
                           session_multiplier: float) -> float:
        """
        Calculate regime-dependent dynamic threshold.
        
        v3.1 Optimization:
        - Medium/High vol: allow trades at lower confidence (0.53-0.55)
        - Low vol: keep stricter confidence (≥0.58)
        - Volatility expansion (atr_ratio > 1.15) further reduces threshold
        """
        # Regime-specific base thresholds
        if regime == 'low':
            threshold = self.threshold_low_vol  # 0.58
        elif regime == 'high':
            threshold = self.threshold_high_vol  # 0.53
        elif regime == 'extreme':
            threshold = self.threshold_extreme_vol  # 0.56
        else:  # medium
            threshold = self.threshold_medium_vol  # 0.55
        
        # Volatility expansion/contraction adjustment
        # Expanding volatility = momentum opportunity = lower threshold
        if atr_ratio > 1.25:
            threshold -= 0.02  # Strong expansion: more aggressive
        elif atr_ratio > 1.15:
            threshold -= 0.01  # Moderate expansion
        elif atr_ratio < 0.75:
            threshold += 0.02  # Strong contraction: more conservative
        elif atr_ratio < 0.85:
            threshold += 0.01  # Moderate contraction
        
        # Apply session multiplier (slight adjustment)
        threshold *= session_multiplier
        
        # Clamp to range (0.53 - 0.60)
        threshold = max(self.min_threshold, min(self.max_threshold, threshold))
        
        return round(threshold, 3)
    
    def check(self, probability: float, threshold: float) -> FilterResult:
        """
        Check if probability exceeds threshold.
        
        Args:
            probability: Model probability (0-1)
            threshold: Dynamic threshold
        """
        # Check for BUY signal
        if probability > threshold:
            return FilterResult(
                passed=True,
                reason=FilterReason.PASSED,
                details={'probability': round(probability, 4), 'threshold': threshold, 'direction': 'BUY'}
            )
        
        # Check for SELL signal
        if probability < (1 - threshold):
            return FilterResult(
                passed=True,
                reason=FilterReason.PASSED,
                details={'probability': round(probability, 4), 'threshold': threshold, 'direction': 'SELL'}
            )
        
        return FilterResult(
            passed=False,
            reason=FilterReason.BELOW_CONFIDENCE,
            details={'probability': round(probability, 4), 'threshold': threshold}
        )


class FlatPredictionFilter:
    """
    Filter 5: Skip flat predictions with regime-adaptive margin.
    
    v3.1 Changes:
    - Default margin: 0.06 (reduced from 0.08)
    - Expanding volatility margin: 0.03 (allows more trades during momentum)
    - Contracting volatility margin: 0.06 (stricter during compression)
    """
    
    def __init__(self, 
                 margin_default: float = 0.06,
                 margin_expanding: float = 0.03,
                 margin_contracting: float = 0.06):
        self.margin_default = margin_default
        self.margin_expanding = margin_expanding
        self.margin_contracting = margin_contracting
    
    def get_adaptive_margin(self, atr_ratio: float) -> float:
        """Get margin based on volatility state."""
        if atr_ratio > 1.15:  # Volatility expanding
            return self.margin_expanding  # 0.03 - more permissive
        elif atr_ratio < 0.85:  # Volatility contracting
            return self.margin_contracting  # 0.06 - stricter
        return self.margin_default  # 0.06
    
    def check(self, probability: float, atr_ratio: float = 1.0) -> FilterResult:
        """
        Check if prediction is too close to 0.5 with adaptive margin.
        
        Args:
            probability: Model probability (0-1)
            atr_ratio: ATR5/ATR20 ratio for adaptive margin
        """
        margin = self.get_adaptive_margin(atr_ratio)
        distance = abs(probability - 0.5)
        passed = distance >= margin
        
        return FilterResult(
            passed=passed,
            reason=FilterReason.PASSED if passed else FilterReason.FLAT_PREDICTION,
            details={
                'probability': round(probability, 4),
                'distance_from_0.5': round(distance, 4),
                'margin': margin,
                'atr_ratio': round(atr_ratio, 3),
                'is_expanding': atr_ratio > 1.15
            }
        )


class ModelAgreementFilter:
    """
    Filter 6: Secondary entry condition via model agreement.
    
    v3.1 Addition:
    Allow trades if both TCN and LightGBM agree on direction,
    even if combined probability is slightly below primary threshold.
    
    Conditions:
    - Both models must predict same direction (both > 0.51 or both < 0.49)
    - Combined probability must be >= 0.52 (floor)
    - Only applies when primary confidence gate fails
    """
    
    def __init__(self, 
                 agreement_threshold: float = 0.51,
                 combined_min: float = 0.52):
        self.agreement_threshold = agreement_threshold
        self.combined_min = combined_min
    
    def check(self, tcn_prob: float, lgbm_prob: float, 
              ensemble_prob: float) -> FilterResult:
        """
        Check if models agree on direction for secondary entry.
        
        Args:
            tcn_prob: TCN model probability
            lgbm_prob: LightGBM model probability
            ensemble_prob: Combined ensemble probability
        """
        # Check if both models agree on BUY
        both_buy = (tcn_prob > self.agreement_threshold and 
                    lgbm_prob > self.agreement_threshold)
        
        # Check if both models agree on SELL
        both_sell = (tcn_prob < (1 - self.agreement_threshold) and 
                     lgbm_prob < (1 - self.agreement_threshold))
        
        # Determine direction
        if both_buy:
            direction = 'BUY'
            agreement = True
        elif both_sell:
            direction = 'SELL'
            agreement = True
        else:
            direction = 'HOLD'
            agreement = False
        
        # Check combined probability floor
        if direction == 'BUY':
            meets_floor = ensemble_prob >= self.combined_min
        elif direction == 'SELL':
            meets_floor = ensemble_prob <= (1 - self.combined_min)
        else:
            meets_floor = False
        
        passed = agreement and meets_floor
        
        return FilterResult(
            passed=passed,
            reason=FilterReason.PASSED_MODEL_AGREEMENT if passed else FilterReason.MODEL_DISAGREEMENT,
            details={
                'tcn_prob': round(tcn_prob, 4),
                'lgbm_prob': round(lgbm_prob, 4),
                'ensemble_prob': round(ensemble_prob, 4),
                'direction': direction,
                'models_agree': agreement,
                'meets_floor': meets_floor,
                'agreement_threshold': self.agreement_threshold,
                'combined_min': self.combined_min
            }
        )


class ExecutionFilterChain:
    """
    Complete filter chain for trade execution (v3.1 Optimized).
    
    BTCUSDm Configuration (v3.1):
    - Spread multiplier: Regime-adaptive (1.8× trending, 2.8× compression)
    - Confidence threshold: Regime-dependent (0.53-0.58)
    - Flat margin: Adaptive (0.03 expanding, 0.06 default)
    - Model agreement: Secondary entry condition
    
    Target: 25-50 trades/day with positive expectancy
    NO SILENT SKIPS - Every decision is logged with full details.
    """
    
    def __init__(self, symbol: str = "BTCUSDm", 
                 spread_mult_trending: float = 1.8,
                 spread_mult_medium: float = 2.2,
                 spread_mult_low: float = 2.8,
                 threshold_low_vol: float = 0.58,
                 threshold_medium_vol: float = 0.55,
                 threshold_high_vol: float = 0.53,
                 threshold_extreme_vol: float = 0.56,
                 flat_margin_default: float = 0.06,
                 flat_margin_expanding: float = 0.03,
                 enable_model_agreement: bool = True,
                 model_agreement_threshold: float = 0.51,
                 model_agreement_combined_min: float = 0.52,
                 # Legacy parameters for backward compatibility
                 spread_multiplier: float = None,
                 base_threshold: float = None,
                 flat_margin: float = None):
        
        self.symbol = symbol.upper()
        self.is_btc = "BTC" in self.symbol
        self.enable_model_agreement = enable_model_agreement
        
        # Initialize filters with v3.1 regime-adaptive settings
        self.spread_filter = SpreadFilter(
            mult_trending=spread_mult_trending,  # 1.8×
            mult_medium=spread_mult_medium,      # 2.2×
            mult_low=spread_mult_low             # 2.8×
        )
        
        self.vol_filter = VolatilityRegimeFilter(
            skip_severe_compression=True,
            compression_threshold=0.7  # Only skip severe compression
        )
        
        self.session_filter = SessionFilter(symbol)
        
        self.confidence_gate = ConfidenceGate(
            threshold_low_vol=threshold_low_vol,      # 0.58
            threshold_medium_vol=threshold_medium_vol,  # 0.55
            threshold_high_vol=threshold_high_vol,    # 0.53
            threshold_extreme_vol=threshold_extreme_vol,  # 0.56
            min_threshold=0.53,
            max_threshold=0.60
        )
        
        self.flat_filter = FlatPredictionFilter(
            margin_default=flat_margin_default,      # 0.06
            margin_expanding=flat_margin_expanding,  # 0.03
            margin_contracting=flat_margin_default   # 0.06
        )
        
        self.model_agreement_filter = ModelAgreementFilter(
            agreement_threshold=model_agreement_threshold,  # 0.51
            combined_min=model_agreement_combined_min       # 0.52
        )
        
        # Stats
        self.total_signals = 0
        self.passed_signals = 0
        self.passed_via_agreement = 0
        self.rejection_counts: Dict[FilterReason, int] = {r: 0 for r in FilterReason}
    
    def execute(self, 
                probability: float,
                expected_move: float,
                spread: float,
                regime: str,
                atr_ratio: float,
                tcn_prob: float = None,
                lgbm_prob: float = None,
                log_rejections: bool = True) -> Tuple[bool, str, Dict]:
        """
        Execute full filter chain with regime-adaptive validation.
        
        ┌─────────────────────────────────────────────────────────────────────┐
        │ REGIME-ADAPTIVE TRADE VALIDATION (v3.1)                             │
        │ Spread multiplier: 1.8× trending, 2.8× compression                  │
        │ Confidence: 0.53-0.58 based on regime                               │
        │ Model agreement: Secondary entry if both models agree               │
        └─────────────────────────────────────────────────────────────────────┘
        
        Args:
            probability: Ensemble probability (0-1)
            expected_move: Expected price move in USD
            spread: Current spread in USD
            regime: Volatility regime ('low', 'medium', 'high', 'extreme')
            atr_ratio: ATR5/ATR20 ratio (>1.0 = expansion, <1.0 = contraction)
            tcn_prob: TCN model probability (for model agreement check)
            lgbm_prob: LightGBM model probability (for model agreement check)
            log_rejections: Whether to log rejection reasons
        
        Returns:
            (passed, signal, details) - NO SILENT SKIPS
        """
        self.total_signals += 1
        details = {
            'signal_id': self.total_signals,
            'probability': round(probability, 4),
            'expected_move': round(expected_move, 4),
            'spread': round(spread, 4),
            'regime': regime,
            'atr_ratio': round(atr_ratio, 3),
            'is_expanding': atr_ratio > 1.15,
            'tcn_prob': round(tcn_prob, 4) if tcn_prob else None,
            'lgbm_prob': round(lgbm_prob, 4) if lgbm_prob else None
        }
        
        # Filter 1: Regime-adaptive spread validation
        result = self.spread_filter.check(expected_move, spread, regime, atr_ratio)
        details['spread_filter'] = result.details
        if not result.passed:
            self._log_rejection(result, log_rejections)
            details['rejection_reason'] = 'spread_too_narrow'
            return False, 'HOLD', details
        
        # Filter 2: Selective volatility regime (only severe compression)
        result = self.vol_filter.check(regime, atr_ratio)
        details['vol_filter'] = result.details
        if not result.passed:
            self._log_rejection(result, log_rejections)
            details['rejection_reason'] = f'volatility_{regime}'
            return False, 'HOLD', details
        
        # Filter 3: Session (get multiplier for threshold adjustment)
        result = self.session_filter.check(strict_mode=False)
        details['session_filter'] = result.details
        session_multiplier = result.details['multiplier']
        
        # Filter 4: Adaptive flat prediction filter
        result = self.flat_filter.check(probability, atr_ratio)
        details['flat_filter'] = result.details
        if not result.passed:
            self._log_rejection(result, log_rejections)
            details['rejection_reason'] = 'flat_prediction'
            return False, 'HOLD', details
        
        # Filter 5: Regime-dependent confidence gate
        threshold = self.confidence_gate.calculate_threshold(regime, atr_ratio, session_multiplier)
        result = self.confidence_gate.check(probability, threshold)
        details['confidence_gate'] = result.details
        details['dynamic_threshold'] = threshold
        
        if result.passed:
            # Primary confidence gate passed
            self.passed_signals += 1
            self.rejection_counts[FilterReason.PASSED] += 1
            
            signal = result.details.get('direction', 'HOLD')
            details['signal'] = signal
            details['rejection_reason'] = None
            details['entry_type'] = 'primary_confidence'
            
            log.info(f"✅ SIGNAL | {self.symbol} | {signal} | Prob: {probability:.1%} | "
                    f"Threshold: {threshold:.1%} | Regime: {regime} | "
                    f"Expected: ${expected_move:.2f} vs Spread: ${spread:.2f}")
            
            return True, signal, details
        
        # Filter 6: Model agreement secondary entry (if primary failed)
        if self.enable_model_agreement and tcn_prob is not None and lgbm_prob is not None:
            agreement_result = self.model_agreement_filter.check(tcn_prob, lgbm_prob, probability)
            details['model_agreement'] = agreement_result.details
            
            if agreement_result.passed:
                # Secondary entry via model agreement
                self.passed_signals += 1
                self.passed_via_agreement += 1
                self.rejection_counts[FilterReason.PASSED_MODEL_AGREEMENT] += 1
                
                signal = agreement_result.details.get('direction', 'HOLD')
                details['signal'] = signal
                details['rejection_reason'] = None
                details['entry_type'] = 'model_agreement'
                
                log.info(f"✅ SIGNAL (Agreement) | {self.symbol} | {signal} | "
                        f"Prob: {probability:.1%} | TCN: {tcn_prob:.1%} | LGBM: {lgbm_prob:.1%} | "
                        f"Regime: {regime}")
                
                return True, signal, details
        
        # All filters failed
        self._log_rejection(result, log_rejections)
        details['rejection_reason'] = 'below_confidence'
        return False, 'HOLD', details
    
    def _log_rejection(self, result: FilterResult, should_log: bool):
        """Log rejection reason."""
        self.rejection_counts[result.reason] += 1
        
        if should_log:
            log.debug(f"FILTER | {self.symbol} | {result}")
    
    def get_stats(self) -> Dict:
        """Get filter statistics."""
        return {
            'total_signals': self.total_signals,
            'passed_signals': self.passed_signals,
            'passed_via_agreement': self.passed_via_agreement,
            'pass_rate': round(self.passed_signals / max(self.total_signals, 1) * 100, 1),
            'agreement_rate': round(self.passed_via_agreement / max(self.passed_signals, 1) * 100, 1),
            'rejections_by_reason': {
                r.value: c for r, c in self.rejection_counts.items() if c > 0
            }
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.total_signals = 0
        self.passed_signals = 0
        self.passed_via_agreement = 0
        self.rejection_counts = {r: 0 for r in FilterReason}
