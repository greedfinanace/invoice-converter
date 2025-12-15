"""
================================================================================
STRATOSPHERE v3.0 - SPREAD-AWARE BACKTESTER
================================================================================
Walk-forward backtesting with:
- Realistic spread modeling
- Commission costs
- Slippage simulation
- Comprehensive metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .config import GLOBAL_CFG, BTC_CFG
from .features import SpreadAwareFeatures
from .models import TCNLightGBMEnsemble
from .filters import ExecutionFilterChain
from .risk import SpreadAwareRiskManager, TPSLLevels

log = logging.getLogger("STRATOSPHERE")


@dataclass
class BacktestTrade:
    """Single backtest trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    lot_size: float
    tp_price: float
    sl_price: float
    exit_reason: str
    gross_pnl: float
    commission: float
    net_pnl: float
    pnl_pips: float


@dataclass
class BacktestResult:
    """Backtest results summary."""
    initial_balance: float
    final_balance: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    trades: List[BacktestTrade]
    equity_curve: List[float]


class SpreadAwareBacktester:
    """
    Realistic backtester with spread and commission modeling.
    BTC ONLY - For Gold use xauusd_hft_bot.py
    """
    
    def __init__(self, 
                 symbol: str = "BTC",
                 initial_balance: float = 10000,
                 commission_per_lot: float = 7.0,
                 slippage_pips: float = 0.5):
        
        self.symbol = symbol.upper()
        
        # BTC ONLY
        if "BTC" not in self.symbol:
            raise ValueError("This backtester is BTC ONLY. For Gold, use xauusd_hft_bot.py")
        
        self.initial_balance = initial_balance
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
        
        # Get config - BTC only
        self.config = BTC_CFG
        
        # Initialize components
        self.features = SpreadAwareFeatures(symbol, GLOBAL_CFG.CORRELATION_THRESHOLD)
        self.model = TCNLightGBMEnsemble(
            tcn_weight=GLOBAL_CFG.TCN_WEIGHT,
            lgbm_weight=GLOBAL_CFG.LGBM_WEIGHT,
            tcn_timesteps=self.config.TCN_TIMESTEPS,
            prediction_horizon=self.config.PREDICTION_HORIZON
        )
        self.filters = ExecutionFilterChain(
            symbol=symbol,
            spread_multiplier=self.config.MIN_EXPECTED_MOVE_MULTIPLIER if self.is_btc else 2.0,
            base_threshold=self.config.BASE_THRESHOLD,
            flat_margin=self.config.FLAT_MARGIN
        )
        self.risk_manager = SpreadAwareRiskManager(symbol)
    
    def run(self, df: pd.DataFrame, train_ratio: float = 0.6) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            df: OHLCV DataFrame
            train_ratio: Ratio of data to use for training
        
        Returns:
            BacktestResult with all metrics
        """
        log.info(f"Running backtest on {len(df)} candles...")
        
        # Split data
        train_size = int(len(df) * train_ratio)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        log.info(f"Train: {len(train_df)} | Test: {len(test_df)}")
        
        # Train model
        X_train, feature_names = self.features.fit_transform(train_df)
        self.model.train(
            X=X_train,
            close=train_df['close'].iloc[-len(X_train):].reset_index(drop=True),
            feature_names=feature_names
        )
        
        # Run backtest
        balance = self.initial_balance
        position = None
        trades: List[BacktestTrade] = []
        equity_curve = [balance]
        
        # Simulate spread (use average of high-low as proxy)
        avg_spread = (test_df['high'] - test_df['low']).mean() * 0.1
        
        timesteps = self.config.TCN_TIMESTEPS
        
        for i in range(timesteps + 10, len(test_df)):
            current_df = test_df.iloc[:i+1]
            current_time = current_df.index[-1]
            current_price = current_df['close'].iloc[-1]
            current_high = current_df['high'].iloc[-1]
            current_low = current_df['low'].iloc[-1]
            
            # Simulate spread
            spread = avg_spread
            
            # Check exit if in position
            if position is not None:
                exit_reason = None
                exit_price = None
                
                if position['direction'] == 'BUY':
                    # Check SL
                    if current_low <= position['sl_price']:
                        exit_reason = 'SL'
                        exit_price = position['sl_price'] - self.slippage_pips * 0.01
                    # Check TP
                    elif current_high >= position['tp_price']:
                        exit_reason = 'TP'
                        exit_price = position['tp_price']
                else:
                    # Check SL
                    if current_high >= position['sl_price']:
                        exit_reason = 'SL'
                        exit_price = position['sl_price'] + self.slippage_pips * 0.01
                    # Check TP
                    elif current_low <= position['tp_price']:
                        exit_reason = 'TP'
                        exit_price = position['tp_price']
                
                if exit_reason:
                    # Calculate P&L
                    if position['direction'] == 'BUY':
                        pnl_pips = (exit_price - position['entry_price']) / 0.01
                    else:
                        pnl_pips = (position['entry_price'] - exit_price) / 0.01
                    
                    gross_pnl = pnl_pips * self.risk_manager.usd_per_pip_per_lot * position['lot_size']
                    commission = self.commission_per_lot * position['lot_size']
                    net_pnl = gross_pnl - commission
                    
                    balance += net_pnl
                    
                    trades.append(BacktestTrade(
                        entry_time=position['entry_time'],
                        exit_time=current_time,
                        direction=position['direction'],
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        lot_size=position['lot_size'],
                        tp_price=position['tp_price'],
                        sl_price=position['sl_price'],
                        exit_reason=exit_reason,
                        gross_pnl=gross_pnl,
                        commission=commission,
                        net_pnl=net_pnl,
                        pnl_pips=pnl_pips
                    ))
                    
                    position = None
            
            # Check entry if no position
            if position is None:
                try:
                    X = self.features.transform(current_df)
                    prediction = self.model.predict(X)
                    
                    # Get regime
                    regime_info = self.features.get_regime(current_df)
                    atr = regime_info.get('atr_5', 0)
                    expected_move = self.risk_manager.get_expected_move(atr, regime_info['regime'])
                    
                    # Execute filters
                    passed, signal, _ = self.filters.execute(
                        probability=prediction['ensemble_prob'],
                        expected_move=expected_move,
                        spread=spread,
                        regime=regime_info['regime'],
                        atr_ratio=regime_info['atr_ratio'],
                        log_rejections=False
                    )
                    
                    if passed and signal in ['BUY', 'SELL']:
                        entry_price = current_price + (spread / 2 if signal == 'BUY' else -spread / 2)
                        entry_price += self.slippage_pips * 0.01 * (1 if signal == 'BUY' else -1)
                        
                        # Calculate TP/SL
                        tp_sl = self.risk_manager.calculate_tp_sl(
                            entry_price=entry_price,
                            direction=signal,
                            atr=atr,
                            regime=regime_info['regime'],
                            spread=spread
                        )
                        
                        # Calculate position size
                        pos_size = self.risk_manager.calculate_position_size(
                            balance=balance,
                            sl_distance=abs(entry_price - tp_sl.sl_price),
                            atr=atr,
                            regime=regime_info['regime']
                        )
                        
                        position = {
                            'direction': signal,
                            'entry_price': entry_price,
                            'entry_time': current_time,
                            'lot_size': pos_size.lot_size,
                            'tp_price': tp_sl.tp_price,
                            'sl_price': tp_sl.sl_price
                        }
                
                except Exception:
                    pass
            
            equity_curve.append(balance)
        
        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve)
    
    def _calculate_metrics(self, trades: List[BacktestTrade], 
                           equity_curve: List[float]) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        
        if not trades:
            return BacktestResult(
                initial_balance=self.initial_balance,
                final_balance=equity_curve[-1] if equity_curve else self.initial_balance,
                total_return_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                profit_factor=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                avg_trade_pnl=0,
                avg_winner=0,
                avg_loser=0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                trades=[],
                equity_curve=equity_curve
            )
        
        final_balance = equity_curve[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100
        
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]
        
        win_rate = len(winners) / len(trades) * 100
        
        gross_profit = sum(t.net_pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.net_pnl for t in losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Max drawdown
        max_dd = 0
        peak = equity_curve[0]
        for e in equity_curve:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            max_dd = max(max_dd, dd)
        
        # Sharpe ratio
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Consecutive wins/losses
        max_cons_wins = max_cons_losses = 0
        cons_wins = cons_losses = 0
        for t in trades:
            if t.net_pnl > 0:
                cons_wins += 1
                cons_losses = 0
                max_cons_wins = max(max_cons_wins, cons_wins)
            else:
                cons_losses += 1
                cons_wins = 0
                max_cons_losses = max(max_cons_losses, cons_losses)
        
        return BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=round(final_balance, 2),
            total_return_pct=round(total_return, 2),
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            avg_trade_pnl=round(sum(t.net_pnl for t in trades) / len(trades), 2),
            avg_winner=round(sum(t.net_pnl for t in winners) / len(winners), 2) if winners else 0,
            avg_loser=round(sum(t.net_pnl for t in losers) / len(losers), 2) if losers else 0,
            max_consecutive_wins=max_cons_wins,
            max_consecutive_losses=max_cons_losses,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def print_results(self, result: BacktestResult):
        """Print backtest results."""
        print("=" * 60)
        print("STRATOSPHERE v3.0 BACKTEST RESULTS")
        print("=" * 60)
        print(f"Symbol:              {self.symbol}")
        print(f"Initial Balance:     ${result.initial_balance:,.2f}")
        print(f"Final Balance:       ${result.final_balance:,.2f}")
        print(f"Total Return:        {result.total_return_pct}%")
        print("-" * 60)
        print(f"Total Trades:        {result.total_trades}")
        print(f"Winning Trades:      {result.winning_trades}")
        print(f"Losing Trades:       {result.losing_trades}")
        print(f"Win Rate:            {result.win_rate}%")
        print(f"Profit Factor:       {result.profit_factor}")
        print("-" * 60)
        print(f"Max Drawdown:        {result.max_drawdown_pct}%")
        print(f"Sharpe Ratio:        {result.sharpe_ratio}")
        print(f"Avg Trade P&L:       ${result.avg_trade_pnl}")
        print(f"Avg Winner:          ${result.avg_winner}")
        print(f"Avg Loser:           ${result.avg_loser}")
        print(f"Max Consec. Wins:    {result.max_consecutive_wins}")
        print(f"Max Consec. Losses:  {result.max_consecutive_losses}")
        print("=" * 60)


def fetch_backtest_data(symbol: str = "GC=F", start: str = "2024-01-01") -> pd.DataFrame:
    """Fetch historical data for backtesting."""
    try:
        import yfinance as yf
        
        log.info(f"Fetching {symbol} data from {start}...")
        df = yf.download(symbol, start=start, interval="1h")
        
        if len(df) == 0:
            log.error("No data available")
            return pd.DataFrame()
        
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        log.info(f"Fetched {len(df)} candles")
        return df
        
    except Exception as e:
        log.error(f"Data fetch failed: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Test backtest - BTC ONLY
    print("STRATOSPHERE v3.0 - BTC BACKTEST MODE")
    print("For Gold backtest, use xauusd_hft_bot.py --backtest")
    
    # Fetch BTC data (using BTC-USD as proxy)
    df = fetch_backtest_data("BTC-USD", "2024-01-01")
    
    if len(df) > 0:
        backtester = SpreadAwareBacktester(
            symbol="BTC",
            initial_balance=10000,
            commission_per_lot=7.0
        )
        
        result = backtester.run(df, train_ratio=0.6)
        backtester.print_results(result)
