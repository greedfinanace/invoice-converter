"""
================================================================================
STRATOSPHERE v3.0 - MONITORING & SAFETY
================================================================================
Continuous monitoring:
- Model probabilities
- Thresholds
- Filter decisions
- Trades skipped vs taken
- Expected vs realized move
- Spread impact per trade

Auto-pause conditions:
- Spread spikes
- Volatility collapse
- Model confidence degradation
"""

import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import threading

log = logging.getLogger("STRATOSPHERE")


@dataclass
class TradeRecord:
    """Record of a single trade for monitoring."""
    timestamp: datetime
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float] = None
    expected_move: float = 0.0
    realized_move: float = 0.0
    spread_at_entry: float = 0.0
    spread_impact_pct: float = 0.0
    model_confidence: float = 0.0
    threshold_used: float = 0.0
    regime: str = "medium"
    pnl: float = 0.0
    exit_reason: str = ""


@dataclass
class MonitoringStats:
    """Aggregated monitoring statistics."""
    # Trade stats
    total_signals: int = 0
    signals_passed: int = 0
    signals_filtered: int = 0
    trades_executed: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    
    # P&L
    gross_pnl: float = 0.0
    total_spread_cost: float = 0.0
    total_commission: float = 0.0
    net_pnl: float = 0.0
    
    # Model performance
    avg_confidence: float = 0.0
    avg_threshold: float = 0.0
    confidence_degradation: bool = False
    
    # Expected vs realized
    avg_expected_move: float = 0.0
    avg_realized_move: float = 0.0
    move_accuracy_pct: float = 0.0
    
    # Filter breakdown
    filter_reasons: Dict[str, int] = field(default_factory=dict)


class StratosphereMonitor:
    """
    Real-time monitoring system for STRATOSPHERE engine.
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 confidence_degradation_threshold: float = 0.10,
                 spread_spike_multiplier: float = 3.0):
        
        self.window_size = window_size
        self.confidence_degradation_threshold = confidence_degradation_threshold
        self.spread_spike_multiplier = spread_spike_multiplier
        
        # Per-symbol tracking
        self.trades: Dict[str, deque] = {}
        self.signals: Dict[str, deque] = {}
        self.spreads: Dict[str, deque] = {}
        self.confidences: Dict[str, deque] = {}
        
        # Global stats
        self.stats: Dict[str, MonitoringStats] = {}
        
        # Alerts
        self.alerts: List[Dict] = []
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def _ensure_symbol(self, symbol: str):
        """Ensure symbol tracking is initialized."""
        if symbol not in self.trades:
            self.trades[symbol] = deque(maxlen=self.window_size)
            self.signals[symbol] = deque(maxlen=self.window_size * 10)
            self.spreads[symbol] = deque(maxlen=self.window_size * 10)
            self.confidences[symbol] = deque(maxlen=self.window_size * 10)
            self.stats[symbol] = MonitoringStats()
    
    def log_signal(self, 
                   symbol: str,
                   probability: float,
                   threshold: float,
                   passed: bool,
                   filter_reason: str,
                   spread: float,
                   regime: str,
                   expected_move: float):
        """Log a signal (passed or filtered)."""
        with self.lock:
            self._ensure_symbol(symbol)
            
            self.signals[symbol].append({
                'timestamp': datetime.now(timezone.utc),
                'probability': probability,
                'threshold': threshold,
                'passed': passed,
                'filter_reason': filter_reason,
                'spread': spread,
                'regime': regime,
                'expected_move': expected_move
            })
            
            self.spreads[symbol].append(spread)
            self.confidences[symbol].append(probability)
            
            # Update stats
            stats = self.stats[symbol]
            stats.total_signals += 1
            if passed:
                stats.signals_passed += 1
            else:
                stats.signals_filtered += 1
                stats.filter_reasons[filter_reason] = stats.filter_reasons.get(filter_reason, 0) + 1
    
    def log_trade_entry(self,
                        symbol: str,
                        direction: str,
                        entry_price: float,
                        spread: float,
                        confidence: float,
                        threshold: float,
                        regime: str,
                        expected_move: float) -> int:
        """Log trade entry. Returns trade ID."""
        with self.lock:
            self._ensure_symbol(symbol)
            
            trade = TradeRecord(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                spread_at_entry=spread,
                model_confidence=confidence,
                threshold_used=threshold,
                regime=regime,
                expected_move=expected_move
            )
            
            self.trades[symbol].append(trade)
            self.stats[symbol].trades_executed += 1
            
            return len(self.trades[symbol]) - 1
    
    def log_trade_exit(self,
                       symbol: str,
                       trade_id: int,
                       exit_price: float,
                       pnl: float,
                       exit_reason: str):
        """Log trade exit."""
        with self.lock:
            self._ensure_symbol(symbol)
            
            if trade_id < len(self.trades[symbol]):
                trade = self.trades[symbol][trade_id]
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.exit_reason = exit_reason
                
                # Calculate realized move
                if trade.direction == 'BUY':
                    trade.realized_move = exit_price - trade.entry_price
                else:
                    trade.realized_move = trade.entry_price - exit_price
                
                # Calculate spread impact
                if trade.realized_move != 0:
                    trade.spread_impact_pct = (trade.spread_at_entry / abs(trade.realized_move)) * 100
                
                # Update stats
                stats = self.stats[symbol]
                if pnl > 0:
                    stats.trades_won += 1
                else:
                    stats.trades_lost += 1
                stats.gross_pnl += pnl
                stats.total_spread_cost += trade.spread_at_entry
    
    def check_safety_conditions(self, symbol: str, current_spread: float) -> Dict:
        """
        Check safety conditions and return alerts.
        
        Returns:
            Dict with 'should_pause', 'reason', 'alerts'
        """
        with self.lock:
            self._ensure_symbol(symbol)
            
            alerts = []
            should_pause = False
            reason = ""
            
            # Check spread spike
            if len(self.spreads[symbol]) >= 10:
                avg_spread = sum(list(self.spreads[symbol])[-10:]) / 10
                if current_spread > avg_spread * self.spread_spike_multiplier:
                    alerts.append({
                        'type': 'SPREAD_SPIKE',
                        'message': f"Spread spike: {current_spread:.4f} vs avg {avg_spread:.4f}",
                        'severity': 'HIGH'
                    })
                    should_pause = True
                    reason = "spread_spike"
            
            # Check confidence degradation
            if len(self.confidences[symbol]) >= 20:
                recent_conf = list(self.confidences[symbol])[-10:]
                older_conf = list(self.confidences[symbol])[-20:-10]
                
                recent_avg = sum(recent_conf) / len(recent_conf)
                older_avg = sum(older_conf) / len(older_conf)
                
                if older_avg - recent_avg > self.confidence_degradation_threshold:
                    alerts.append({
                        'type': 'CONFIDENCE_DEGRADATION',
                        'message': f"Confidence dropped: {older_avg:.2%} -> {recent_avg:.2%}",
                        'severity': 'MEDIUM'
                    })
                    self.stats[symbol].confidence_degradation = True
            
            # Check win rate degradation
            stats = self.stats[symbol]
            if stats.trades_executed >= 20:
                win_rate = stats.trades_won / stats.trades_executed
                if win_rate < 0.40:
                    alerts.append({
                        'type': 'LOW_WIN_RATE',
                        'message': f"Win rate below threshold: {win_rate:.1%}",
                        'severity': 'MEDIUM'
                    })
            
            self.alerts.extend(alerts)
            
            return {
                'should_pause': should_pause,
                'reason': reason,
                'alerts': alerts
            }
    
    def get_stats(self, symbol: str) -> MonitoringStats:
        """Get monitoring stats for symbol."""
        with self.lock:
            self._ensure_symbol(symbol)
            
            stats = self.stats[symbol]
            
            # Calculate averages
            if self.confidences[symbol]:
                stats.avg_confidence = sum(self.confidences[symbol]) / len(self.confidences[symbol])
            
            if self.signals[symbol]:
                thresholds = [s['threshold'] for s in self.signals[symbol]]
                stats.avg_threshold = sum(thresholds) / len(thresholds)
                
                expected = [s['expected_move'] for s in self.signals[symbol]]
                stats.avg_expected_move = sum(expected) / len(expected)
            
            # Calculate realized move average
            trades_with_exit = [t for t in self.trades[symbol] if t.exit_price is not None]
            if trades_with_exit:
                stats.avg_realized_move = sum(t.realized_move for t in trades_with_exit) / len(trades_with_exit)
                
                if stats.avg_expected_move > 0:
                    stats.move_accuracy_pct = (stats.avg_realized_move / stats.avg_expected_move) * 100
            
            return stats
    
    def print_summary(self, symbol: str):
        """Print monitoring summary for symbol."""
        stats = self.get_stats(symbol)
        
        win_rate = stats.trades_won / max(stats.trades_executed, 1) * 100
        filter_rate = stats.signals_filtered / max(stats.total_signals, 1) * 100
        
        log.info("=" * 60)
        log.info(f"MONITORING SUMMARY: {symbol}")
        log.info("=" * 60)
        log.info(f"Signals: {stats.total_signals} | Passed: {stats.signals_passed} | Filtered: {stats.signals_filtered} ({filter_rate:.1f}%)")
        log.info(f"Trades: {stats.trades_executed} | Won: {stats.trades_won} | Lost: {stats.trades_lost} | Win Rate: {win_rate:.1f}%")
        log.info(f"P&L: ${stats.gross_pnl:.2f} | Spread Cost: ${stats.total_spread_cost:.2f}")
        log.info(f"Avg Confidence: {stats.avg_confidence:.2%} | Avg Threshold: {stats.avg_threshold:.3f}")
        log.info(f"Expected Move: {stats.avg_expected_move:.4f} | Realized: {stats.avg_realized_move:.4f} | Accuracy: {stats.move_accuracy_pct:.1f}%")
        
        if stats.filter_reasons:
            log.info("Filter Breakdown:")
            for reason, count in sorted(stats.filter_reasons.items(), key=lambda x: -x[1]):
                log.info(f"  {reason}: {count}")
        
        if stats.confidence_degradation:
            log.warning("⚠️ CONFIDENCE DEGRADATION DETECTED")
        
        log.info("=" * 60)
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts."""
        with self.lock:
            return self.alerts[-limit:]
    
    def clear_alerts(self):
        """Clear all alerts."""
        with self.lock:
            self.alerts.clear()


class TradeLogger:
    """CSV trade logger for audit trail."""
    
    def __init__(self, filepath: str = "stratosphere_v3_trades.csv"):
        self.filepath = filepath
        self._init_file()
    
    def _init_file(self):
        """Initialize CSV file with headers."""
        import os
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                f.write("timestamp,symbol,direction,entry_price,exit_price,lot_size,"
                       "tp_price,sl_price,exit_reason,gross_pnl,commission,net_pnl,"
                       "spread,confidence,threshold,regime,expected_move,realized_move\n")
    
    def log_trade(self, trade: TradeRecord, lot_size: float = 0.01, commission: float = 0.0):
        """Log trade to CSV."""
        with open(self.filepath, 'a') as f:
            f.write(f"{trade.timestamp.isoformat()},{trade.symbol},{trade.direction},"
                   f"{trade.entry_price},{trade.exit_price or ''},{lot_size},"
                   f"{0},{0},{trade.exit_reason},{trade.pnl},{commission},{trade.pnl - commission},"
                   f"{trade.spread_at_entry},{trade.model_confidence},{trade.threshold_used},"
                   f"{trade.regime},{trade.expected_move},{trade.realized_move}\n")
