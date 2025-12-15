"""
================================================================================
STRATOSPHERE v3.1 - ASYMMETRIC SPREAD-AWARE RISK MANAGEMENT
================================================================================
BTCUSDm 5-Minute Momentum Mode (Optimized for Trade Density)

┌─────────────────────────────────────────────────────────────────────┐
│ REGIME-ADAPTIVE TRADE VALIDATION (v3.1)                             │
│ Volatility-scaled spread: 1.8× trending, 2.8× compression           │
│ Asymmetric TP/SL: Wider TP in expansion, Tighter SL at entry        │
└─────────────────────────────────────────────────────────────────────┘

BTC Configuration (v3.1):
- Typical spread: ~$18
- Max spread: $22 (pause if exceeded)
- TP: $45 – $180 (asymmetric: wider in expansion)
- SL: $25 – $60 (tighter at entry for higher win-rate)
- R:R minimum = 1.3:1 (1.5:1 in trending)
- Volatility-scaled reward expectation (1.8× spread in trending)

Key Changes:
- Asymmetric TP/SL optimization
- Regime-adaptive spread multipliers
- Tighter SL at entry to increase win-rate
- Wider TP during volatility expansion
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

log = logging.getLogger("STRATOSPHERE")


class ExitType(Enum):
    """Types of position exits."""
    TAKE_PROFIT = "TP"
    STOP_LOSS = "SL"
    TRAILING_STOP = "TRAIL"
    PARTIAL_EXIT = "PARTIAL"
    BREAKEVEN = "BE"
    TIME_EXIT = "TIME"
    SIGNAL_EXIT = "SIGNAL"


@dataclass
class TPSLLevels:
    """Take profit and stop loss levels."""
    tp_price: float
    sl_price: float
    tp_pips: float
    sl_pips: float
    risk_reward: float
    partial_tp: Optional[float] = None
    breakeven_trigger: Optional[float] = None


@dataclass
class PositionSize:
    """Position sizing result."""
    lot_size: float
    risk_amount: float
    max_loss: float
    position_value: float


class SpreadAwareRiskManager:
    """
    Asymmetric spread-aware risk management for BTC (v3.1).
    
    Key principles:
    1. Regime-adaptive spread validation (1.8× trending, 2.8× compression)
    2. Asymmetric TP/SL: Wider TP in expansion, Tighter SL at entry
    3. Dynamic R:R based on regime (1.3 base, 1.5 trending)
    4. Volatility-adjusted position sizing
    """
    
    def __init__(self, symbol: str = "BTCUSDm"):
        self.symbol = symbol.upper()
        self.is_btc = "BTC" in self.symbol
        
        # BTC settings (v3.1 - optimized for trade density)
        if self.is_btc:
            # Asymmetric TP/SL ranges
            self.tp_min = 45.0         # Lowered from 60 (allows more trades)
            self.tp_max = 180.0        # Increased from 150 (wider in expansion)
            self.sl_min = 25.0         # Tighter SL (lowered from 40)
            self.sl_max = 60.0         # Reduced from 80 (tighter risk)
            
            # Regime-adaptive R:R
            self.min_rr = 1.3          # Lowered from 1.5 (allows more trades)
            self.min_rr_trending = 1.5  # Higher R:R in trending markets
            
            # Asymmetric multipliers
            self.tp_mult_expansion = 1.4   # Wider TP in volatility expansion
            self.tp_mult_contraction = 0.9  # Tighter TP in contraction
            self.sl_mult_entry = 0.8       # Tighter SL at entry (increase win-rate)
            
            # Spread requirements (regime-adaptive)
            self.typical_spread = 18.0
            self.max_spread = 22.0      # Increased from 20
            
            # Regime-adaptive spread multipliers
            self.spread_mult_trending = 1.8   # 1.8× in trending (vs 3×)
            self.spread_mult_medium = 2.2     # 2.2× in medium vol
            self.spread_mult_low = 2.8        # 2.8× in low vol
            self.min_expected_move = 32.0     # 1.8 × $18 = $32.4
            
            self.pip_value = 1.0
            self.usd_per_pip_per_lot = 1.0
            self.allow_micro_scalps = False
        else:
            # XAU settings (separate system - xauusd_hft_bot.py)
            self.tp_min = 0.10
            self.tp_max = 0.50
            self.sl_min = 0.05
            self.sl_max = 0.30
            self.min_rr = 1.2
            self.min_rr_trending = 1.4
            self.tp_mult_expansion = 1.3
            self.tp_mult_contraction = 0.9
            self.sl_mult_entry = 0.85
            self.typical_spread = 0.02
            self.spread_mult_trending = 1.5
            self.spread_mult_medium = 2.0
            self.spread_mult_low = 2.5
            self.min_expected_move = 0.03
            self.pip_value = 0.01
            self.usd_per_pip_per_lot = 10.0
            self.allow_micro_scalps = True
        
        # Risk parameters
        self.risk_per_trade_pct = 0.02
        self.max_position_pct = 0.50
        
        # Partial exit settings
        self.partial_exit_pct = 0.50
        self.partial_tp_ratio = 0.5
        self.breakeven_trigger_ratio = 0.4
    
    def calculate_tp_sl(self, 
                        entry_price: float,
                        direction: str,
                        atr: float,
                        regime: str,
                        spread: float,
                        atr_ratio: float = 1.0) -> TPSLLevels:
        """
        Calculate asymmetric TP/SL levels based on ATR and regime (v3.1).
        
        Asymmetric Optimization:
        - Wider TP during volatility expansion (trend continuation)
        - Tighter SL during entry to increase win-rate
        
        Args:
            entry_price: Entry price
            direction: 'BUY' or 'SELL'
            atr: Current ATR value
            regime: Volatility regime ('low', 'medium', 'high', 'extreme')
            spread: Current spread
            atr_ratio: ATR5/ATR20 ratio (>1.0 = expansion)
        
        Returns:
            TPSLLevels with all calculated levels
        """
        # Determine if volatility is expanding or contracting
        is_expanding = atr_ratio > 1.15
        is_contracting = atr_ratio < 0.85
        is_trending = regime in ['high', 'extreme'] or is_expanding
        
        # Base multipliers by regime
        if regime == 'low':
            sl_mult = 1.0   # Tighter SL in low vol
            tp_mult = 1.6   # Moderate TP
        elif regime == 'high':
            sl_mult = 1.2   # Moderate SL
            tp_mult = 2.2   # Wider TP for momentum
        elif regime == 'extreme':
            sl_mult = 1.5   # Wider SL for protection
            tp_mult = 2.8   # Wide TP for big moves
        else:  # medium
            sl_mult = 1.1
            tp_mult = 1.9
        
        # Apply asymmetric adjustments based on volatility state
        if is_expanding:
            # Volatility expansion: wider TP (trend continuation)
            tp_mult *= self.tp_mult_expansion  # 1.4×
            # Keep SL tight at entry for better win-rate
            sl_mult *= self.sl_mult_entry  # 0.8×
        elif is_contracting:
            # Volatility contraction: tighter TP
            tp_mult *= self.tp_mult_contraction  # 0.9×
            # Tighter SL in compression
            sl_mult *= 0.9
        
        # Calculate raw levels
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult
        
        # Clamp to min/max
        sl_distance = max(self.sl_min, min(self.sl_max, sl_distance))
        tp_distance = max(self.tp_min, min(self.tp_max, tp_distance))
        
        # Ensure minimum R:R (regime-adaptive)
        min_rr = self.min_rr_trending if is_trending else self.min_rr
        if tp_distance / max(sl_distance, 0.01) < min_rr:
            tp_distance = sl_distance * min_rr
            tp_distance = min(self.tp_max, tp_distance)
        
        # Ensure TP exceeds spread (regime-adaptive multiplier)
        if not self.allow_micro_scalps:
            if is_trending:
                min_tp = spread * self.spread_mult_trending  # 1.8×
            elif regime == 'low':
                min_tp = spread * self.spread_mult_low  # 2.8×
            else:
                min_tp = spread * self.spread_mult_medium  # 2.2×
            tp_distance = max(tp_distance, min_tp)
        
        # Calculate actual prices
        if direction == 'BUY':
            tp_price = entry_price + tp_distance
            sl_price = entry_price - sl_distance
            partial_tp = entry_price + (tp_distance * self.partial_tp_ratio)
            breakeven_trigger = entry_price + (tp_distance * self.breakeven_trigger_ratio)
        else:
            tp_price = entry_price - tp_distance
            sl_price = entry_price + sl_distance
            partial_tp = entry_price - (tp_distance * self.partial_tp_ratio)
            breakeven_trigger = entry_price - (tp_distance * self.breakeven_trigger_ratio)
        
        return TPSLLevels(
            tp_price=round(tp_price, 2),
            sl_price=round(sl_price, 2),
            tp_pips=round(tp_distance / self.pip_value, 1),
            sl_pips=round(sl_distance / self.pip_value, 1),
            risk_reward=round(tp_distance / max(sl_distance, 0.01), 2),
            partial_tp=round(partial_tp, 2) if self.allow_micro_scalps else None,
            breakeven_trigger=round(breakeven_trigger, 2)
        )
    
    def calculate_position_size(self,
                                balance: float,
                                sl_distance: float,
                                atr: float,
                                regime: str) -> PositionSize:
        """
        Calculate position size based on risk and volatility.
        
        Args:
            balance: Account balance
            sl_distance: Stop loss distance in price units
            atr: Current ATR
            regime: Volatility regime
        
        Returns:
            PositionSize with lot size and risk metrics
        """
        # Base risk amount
        risk_amount = balance * self.risk_per_trade_pct
        
        # Calculate lot size
        sl_pips = sl_distance / self.pip_value
        lot_size = risk_amount / (sl_pips * self.usd_per_pip_per_lot)
        
        # Volatility adjustment (reduce size in extreme volatility)
        if regime == 'extreme':
            lot_size *= 0.5
        elif regime == 'high':
            lot_size *= 0.75
        
        # Apply limits
        lot_size = max(0.01, min(round(lot_size, 2), 10.0))
        
        # Check max exposure
        max_lot = (balance * self.max_position_pct) / (sl_pips * self.usd_per_pip_per_lot)
        lot_size = min(lot_size, max_lot)
        lot_size = round(lot_size, 2)
        
        # Calculate actual risk
        max_loss = sl_pips * self.usd_per_pip_per_lot * lot_size
        position_value = lot_size * 100000  # Standard lot = 100,000 units
        
        return PositionSize(
            lot_size=lot_size,
            risk_amount=round(risk_amount, 2),
            max_loss=round(max_loss, 2),
            position_value=round(position_value, 2)
        )
    
    def check_exit_conditions(self,
                              entry_price: float,
                              current_price: float,
                              direction: str,
                              tp_sl: TPSLLevels,
                              entry_time_seconds: float = 0) -> Tuple[bool, Optional[ExitType], Dict]:
        """
        Check if any exit condition is met.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            direction: 'BUY' or 'SELL'
            tp_sl: TP/SL levels
            entry_time_seconds: Seconds since entry
        
        Returns:
            (should_exit, exit_type, details)
        """
        details = {}
        
        # Calculate current P&L
        if direction == 'BUY':
            pnl_pips = (current_price - entry_price) / self.pip_value
            hit_tp = current_price >= tp_sl.tp_price
            hit_sl = current_price <= tp_sl.sl_price
            hit_partial = tp_sl.partial_tp and current_price >= tp_sl.partial_tp
            hit_be_trigger = tp_sl.breakeven_trigger and current_price >= tp_sl.breakeven_trigger
        else:
            pnl_pips = (entry_price - current_price) / self.pip_value
            hit_tp = current_price <= tp_sl.tp_price
            hit_sl = current_price >= tp_sl.sl_price
            hit_partial = tp_sl.partial_tp and current_price <= tp_sl.partial_tp
            hit_be_trigger = tp_sl.breakeven_trigger and current_price <= tp_sl.breakeven_trigger
        
        details['pnl_pips'] = round(pnl_pips, 1)
        details['current_price'] = current_price
        
        # Check TP
        if hit_tp:
            return True, ExitType.TAKE_PROFIT, details
        
        # Check SL
        if hit_sl:
            return True, ExitType.STOP_LOSS, details
        
        # Check partial exit (for XAU)
        if hit_partial and self.allow_micro_scalps:
            details['partial_exit'] = True
            return True, ExitType.PARTIAL_EXIT, details
        
        # Check breakeven trigger
        if hit_be_trigger:
            details['move_to_breakeven'] = True
        
        # Time-based exit for losers (optional)
        if entry_time_seconds > 180 and pnl_pips < 0:  # 3 minutes
            details['time_exit'] = True
            return True, ExitType.TIME_EXIT, details
        
        return False, None, details
    
    def get_expected_move(self, atr: float, regime: str, atr_ratio: float = 1.0) -> float:
        """
        Estimate expected price move based on ATR and regime (v3.1).
        Used for spread filter validation.
        
        For BTC momentum mode:
        - Regime-adaptive: 1.8× spread in trending, 2.8× in compression
        - Volatility expansion increases expected move estimate
        """
        # Base multiplier by regime
        if regime == 'low':
            base_mult = 0.6
        elif regime == 'high':
            base_mult = 1.6
        elif regime == 'extreme':
            base_mult = 2.2
        else:  # medium
            base_mult = 1.1
        
        # Adjust for volatility expansion/contraction
        if atr_ratio > 1.25:
            # Strong expansion: increase expected move
            base_mult *= 1.3
        elif atr_ratio > 1.15:
            # Moderate expansion
            base_mult *= 1.15
        elif atr_ratio < 0.75:
            # Strong contraction: decrease expected move
            base_mult *= 0.7
        elif atr_ratio < 0.85:
            # Moderate contraction
            base_mult *= 0.85
        
        return atr * base_mult
    
    def get_spread_multiplier(self, regime: str, atr_ratio: float = 1.0) -> float:
        """
        Get regime-adaptive spread multiplier for validation.
        
        Returns:
            1.8× for trending/expanding
            2.2× for medium
            2.8× for low/contracting
        """
        is_expanding = atr_ratio > 1.15
        is_contracting = atr_ratio < 0.85
        
        if is_expanding or regime in ['high', 'extreme']:
            return self.spread_mult_trending  # 1.8×
        elif regime == 'low' or is_contracting:
            return self.spread_mult_low  # 2.8×
        return self.spread_mult_medium  # 2.2×
    
    def validate_spread_economics(self, expected_move: float, spread: float,
                                   regime: str = 'medium', atr_ratio: float = 1.0) -> bool:
        """
        Validate that expected move exceeds regime-adaptive spread threshold.
        
        v3.1: Uses 1.8× in trending, 2.8× in compression (vs hard 3×)
        """
        multiplier = self.get_spread_multiplier(regime, atr_ratio)
        min_required = spread * multiplier
        return expected_move >= min_required
    
    def should_move_to_breakeven(self,
                                  entry_price: float,
                                  current_price: float,
                                  direction: str,
                                  tp_sl: TPSLLevels) -> bool:
        """Check if SL should be moved to breakeven."""
        if tp_sl.breakeven_trigger is None:
            return False
        
        if direction == 'BUY':
            return current_price >= tp_sl.breakeven_trigger
        else:
            return current_price <= tp_sl.breakeven_trigger
    
    def calculate_trailing_stop(self,
                                 entry_price: float,
                                 current_price: float,
                                 direction: str,
                                 current_sl: float,
                                 trail_distance: float) -> Optional[float]:
        """
        Calculate new trailing stop level.
        
        Returns new SL price if it should be updated, None otherwise.
        """
        if direction == 'BUY':
            new_sl = current_price - trail_distance
            if new_sl > current_sl and new_sl > entry_price:
                return round(new_sl, 2)
        else:
            new_sl = current_price + trail_distance
            if new_sl < current_sl and new_sl < entry_price:
                return round(new_sl, 2)
        
        return None


class VirtualOrderManager:
    """
    Virtual Order Management System (VOMS).
    Manages SL/TP in memory to reduce API calls.
    """
    
    def __init__(self, risk_manager: SpreadAwareRiskManager):
        self.risk_manager = risk_manager
        self.positions: Dict[int, Dict] = {}
    
    def add_position(self,
                     ticket: int,
                     symbol: str,
                     direction: str,
                     entry_price: float,
                     lot_size: float,
                     tp_sl: TPSLLevels) -> Dict:
        """Register new position with virtual SL/TP."""
        import time
        
        position = {
            'ticket': ticket,
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'lot_size': lot_size,
            'tp_price': tp_sl.tp_price,
            'sl_price': tp_sl.sl_price,
            'partial_tp': tp_sl.partial_tp,
            'breakeven_trigger': tp_sl.breakeven_trigger,
            'entry_time': time.time(),
            'trailing_active': False,
            'partial_closed': False,
            'at_breakeven': False
        }
        
        self.positions[ticket] = position
        
        log.info(f"VOMS | {symbol} | {direction} {lot_size} @ {entry_price} | "
                f"TP: {tp_sl.tp_price} | SL: {tp_sl.sl_price} | R:R: {tp_sl.risk_reward}")
        
        return position
    
    def check_exits(self, get_price_func) -> list:
        """
        Check all positions for exit conditions.
        
        Args:
            get_price_func: Function(symbol) -> (bid, ask, spread)
        
        Returns:
            List of (ticket, exit_type, details) for positions to close
        """
        import time
        
        exits = []
        
        for ticket, pos in list(self.positions.items()):
            bid, ask, spread = get_price_func(pos['symbol'])
            if bid == 0:
                continue
            
            current_price = bid if pos['direction'] == 'BUY' else ask
            entry_time_seconds = time.time() - pos['entry_time']
            
            # Create TPSLLevels from position
            tp_sl = TPSLLevels(
                tp_price=pos['tp_price'],
                sl_price=pos['sl_price'],
                tp_pips=0,  # Not needed for exit check
                sl_pips=0,
                risk_reward=0,
                partial_tp=pos['partial_tp'],
                breakeven_trigger=pos['breakeven_trigger']
            )
            
            should_exit, exit_type, details = self.risk_manager.check_exit_conditions(
                pos['entry_price'],
                current_price,
                pos['direction'],
                tp_sl,
                entry_time_seconds
            )
            
            if should_exit:
                exits.append((ticket, exit_type, details))
            elif details.get('move_to_breakeven') and not pos['at_breakeven']:
                # Update SL to breakeven
                pos['sl_price'] = pos['entry_price']
                pos['at_breakeven'] = True
                log.info(f"VOMS | {pos['symbol']} | Moved SL to breakeven @ {pos['entry_price']}")
        
        return exits
    
    def remove_position(self, ticket: int):
        """Remove closed position."""
        if ticket in self.positions:
            del self.positions[ticket]
    
    def get_position_count(self, symbol: str = None) -> int:
        """Get number of open positions."""
        if symbol:
            return sum(1 for p in self.positions.values() if p['symbol'] == symbol)
        return len(self.positions)
