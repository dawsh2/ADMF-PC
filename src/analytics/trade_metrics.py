"""
Trade Metrics for ADMF-PC

Utilities for analyzing trades from sparse signal storage.
Reconstructs full trading activity from signal changes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    strategy_id: int
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    direction: str  # 'long' or 'short'
    entry_signal: float
    exit_signal: float
    
    @property
    def duration(self) -> int:
        """Trade duration in bars."""
        return self.exit_bar - self.entry_bar
    
    @property
    def pnl(self) -> float:
        """Profit/loss in points."""
        if self.direction == 'long':
            return self.exit_price - self.entry_price
        else:
            return self.entry_price - self.exit_price
    
    @property
    def return_pct(self) -> float:
        """Return percentage using signal-based formula."""
        # Use signal value for clean calculation: signal * (exit - entry) / entry
        return self.entry_signal * (self.exit_price - self.entry_price) / self.entry_price
    
    @property
    def is_winner(self) -> bool:
        """Whether trade was profitable."""
        return self.pnl > 0


class TradeAnalyzer:
    """
    Analyzes trades from sparse signal storage.
    
    Handles signal pairing, trade extraction, and performance calculation.
    """
    
    def __init__(self, signals_df: pd.DataFrame):
        """
        Initialize with signals DataFrame.
        
        Args:
            signals_df: DataFrame with columns [strategy_id, bar_idx, signal_value, price]
        """
        self.signals_df = signals_df.sort_values(['strategy_id', 'bar_idx'])
        self.trades: List[Trade] = []
        self._extract_trades()
    
    def _extract_trades(self):
        """Extract trades from signal transitions."""
        self.trades = []
        
        for strategy_id in self.signals_df['strategy_id'].unique():
            strategy_signals = self.signals_df[
                self.signals_df['strategy_id'] == strategy_id
            ].copy()
            
            # Track open position
            position = None
            entry_bar = None
            entry_price = None
            entry_signal = None
            
            for _, row in strategy_signals.iterrows():
                signal = row['signal_value']
                
                if position is None and signal != 0:
                    # Open new position
                    position = 'long' if signal > 0 else 'short'
                    entry_bar = row['bar_idx']
                    entry_price = row['price']
                    entry_signal = signal
                
                elif position is not None:
                    # Check for exit conditions
                    should_exit = False
                    
                    if signal == 0:
                        # Explicit exit signal
                        should_exit = True
                    elif (position == 'long' and signal < 0) or (position == 'short' and signal > 0):
                        # Reversal - exit current and open new
                        should_exit = True
                    
                    if should_exit:
                        # Create trade
                        trade = Trade(
                            strategy_id=strategy_id,
                            entry_bar=entry_bar,
                            exit_bar=row['bar_idx'],
                            entry_price=entry_price,
                            exit_price=row['price'],
                            direction=position,
                            entry_signal=entry_signal,
                            exit_signal=signal
                        )
                        self.trades.append(trade)
                        
                        # Handle reversal
                        if signal != 0:
                            position = 'long' if signal > 0 else 'short'
                            entry_bar = row['bar_idx']
                            entry_price = row['price']
                            entry_signal = signal
                        else:
                            position = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades:
            data.append({
                'strategy_id': trade.strategy_id,
                'entry_bar': trade.entry_bar,
                'exit_bar': trade.exit_bar,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'direction': trade.direction,
                'duration': trade.duration,
                'pnl': trade.pnl,
                'return_pct': trade.return_pct,
                'is_winner': trade.is_winner
            })
        
        return pd.DataFrame(data)
    
    def summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics for all trades."""
        if not self.trades:
            return {'total_trades': 0}
        
        df = self.to_dataframe()
        
        return {
            'total_trades': len(self.trades),
            'win_rate': df['is_winner'].mean(),
            'avg_return': df['return_pct'].mean(),
            'avg_winner': df[df['is_winner']]['return_pct'].mean(),
            'avg_loser': df[~df['is_winner']]['return_pct'].mean(),
            'best_trade': df['return_pct'].max(),
            'worst_trade': df['return_pct'].min(),
            'avg_duration': df['duration'].mean(),
            'total_pnl': df['pnl'].sum(),
            'profit_factor': self._calculate_profit_factor(df),
            'sharpe_ratio': self._calculate_sharpe(df),
            'max_drawdown': self._calculate_max_drawdown(df)
        }
    
    def by_strategy(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics grouped by strategy."""
        results = {}
        df = self.to_dataframe()
        
        for strategy_id in df['strategy_id'].unique():
            strategy_df = df[df['strategy_id'] == strategy_id]
            
            results[strategy_id] = {
                'trades': len(strategy_df),
                'win_rate': strategy_df['is_winner'].mean(),
                'avg_return': strategy_df['return_pct'].mean(),
                'sharpe_ratio': self._calculate_sharpe(strategy_df),
                'profit_factor': self._calculate_profit_factor(strategy_df),
                'avg_duration': strategy_df['duration'].mean()
            }
        
        return results
    
    def _calculate_profit_factor(self, df: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if df.empty:
            return 0.0
        
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_sharpe(self, df: pd.DataFrame, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        if df.empty or len(df) < 2:
            return 0.0
        
        returns = df['return_pct'].values
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return * np.sqrt(periods_per_year)
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        if df.empty:
            return 0.0
        
        # Calculate cumulative returns
        df = df.sort_values('exit_bar')
        cum_returns = (1 + df['return_pct']).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdowns
        drawdowns = (cum_returns - running_max) / running_max
        
        return drawdowns.min()


def reconstruct_equity_curve(signals_df: pd.DataFrame, 
                           initial_capital: float = 100000,
                           position_size: float = 1.0) -> pd.DataFrame:
    """
    Reconstruct full equity curve from sparse signals.
    
    Args:
        signals_df: DataFrame with signal data
        initial_capital: Starting capital
        position_size: Position size (shares/contracts)
        
    Returns:
        DataFrame with equity curve
    """
    # Ensure sorted by time
    signals_df = signals_df.sort_values(['strategy_id', 'bar_idx'])
    
    equity_curves = []
    
    for strategy_id in signals_df['strategy_id'].unique():
        strategy_signals = signals_df[signals_df['strategy_id'] == strategy_id]
        
        # Initialize tracking variables
        equity = initial_capital
        position = 0
        entry_price = 0
        
        equity_data = []
        
        # Get all unique bar indices
        all_bars = range(
            strategy_signals['bar_idx'].min(),
            strategy_signals['bar_idx'].max() + 1
        )
        
        # Create signal lookup
        signal_lookup = dict(zip(
            strategy_signals['bar_idx'],
            strategy_signals['signal_value']
        ))
        price_lookup = dict(zip(
            strategy_signals['bar_idx'],
            strategy_signals['price']
        ))
        
        for bar in all_bars:
            signal = signal_lookup.get(bar, 0)
            price = price_lookup.get(bar)
            
            # Handle position changes
            if signal != 0 and position == 0:
                # Enter position
                position = position_size if signal > 0 else -position_size
                entry_price = price
            
            elif signal == 0 and position != 0:
                # Exit position
                if price:
                    pnl = position * (price - entry_price)
                    equity += pnl
                position = 0
            
            elif signal != 0 and position != 0:
                # Check for reversal
                new_position = position_size if signal > 0 else -position_size
                if np.sign(new_position) != np.sign(position):
                    # Close and reverse
                    if price:
                        pnl = position * (price - entry_price)
                        equity += pnl
                    position = new_position
                    entry_price = price
            
            # Record equity
            equity_data.append({
                'strategy_id': strategy_id,
                'bar_idx': bar,
                'equity': equity,
                'position': position,
                'signal': signal,
                'price': price
            })
        
        equity_curves.append(pd.DataFrame(equity_data))
    
    if equity_curves:
        return pd.concat(equity_curves, ignore_index=True)
    else:
        return pd.DataFrame()


def calculate_rolling_metrics(trades_df: pd.DataFrame,
                            window_size: int = 50,
                            min_trades: int = 10) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Args:
        trades_df: DataFrame of trades
        window_size: Rolling window size (number of trades)
        min_trades: Minimum trades for calculation
        
    Returns:
        DataFrame with rolling metrics
    """
    if len(trades_df) < min_trades:
        return pd.DataFrame()
    
    # Sort by exit time
    trades_df = trades_df.sort_values('exit_bar')
    
    # Calculate rolling metrics
    rolling_data = []
    
    for i in range(min_trades, len(trades_df) + 1):
        start_idx = max(0, i - window_size)
        window = trades_df.iloc[start_idx:i]
        
        metrics = {
            'exit_bar': window.iloc[-1]['exit_bar'],
            'trades_in_window': len(window),
            'win_rate': window['is_winner'].mean(),
            'avg_return': window['return_pct'].mean(),
            'sharpe': TradeAnalyzer([])._calculate_sharpe(window),
            'profit_factor': TradeAnalyzer([])._calculate_profit_factor(window)
        }
        
        rolling_data.append(metrics)
    
    return pd.DataFrame(rolling_data)


def analyze_trade_clustering(trades_df: pd.DataFrame,
                           time_threshold: int = 10) -> Dict[str, Any]:
    """
    Analyze clustering of trades in time.
    
    Args:
        trades_df: DataFrame of trades
        time_threshold: Bars between trades to consider clustered
        
    Returns:
        Analysis of trade clustering patterns
    """
    if trades_df.empty:
        return {'clusters': 0}
    
    # Sort by entry time
    trades_df = trades_df.sort_values('entry_bar')
    
    clusters = []
    current_cluster = []
    
    for _, trade in trades_df.iterrows():
        if not current_cluster:
            current_cluster.append(trade)
        else:
            # Check if this trade is close to the last one
            last_exit = current_cluster[-1]['exit_bar']
            if trade['entry_bar'] - last_exit <= time_threshold:
                current_cluster.append(trade)
            else:
                # Start new cluster
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [trade]
    
    # Don't forget the last cluster
    if len(current_cluster) > 1:
        clusters.append(current_cluster)
    
    # Analyze clusters
    cluster_analysis = {
        'total_clusters': len(clusters),
        'avg_cluster_size': np.mean([len(c) for c in clusters]) if clusters else 0,
        'max_cluster_size': max([len(c) for c in clusters]) if clusters else 0,
        'clustered_trades': sum(len(c) for c in clusters),
        'clustering_ratio': sum(len(c) for c in clusters) / len(trades_df) if trades_df.size else 0
    }
    
    # Performance in clusters vs isolated trades
    if clusters:
        clustered_returns = []
        for cluster in clusters:
            for trade in cluster:
                clustered_returns.append(trade['return_pct'])
        
        isolated_trades = trades_df[
            ~trades_df.index.isin([t.name for c in clusters for t in c])
        ]
        
        cluster_analysis['clustered_avg_return'] = np.mean(clustered_returns) if clustered_returns else 0
        cluster_analysis['isolated_avg_return'] = isolated_trades['return_pct'].mean() if not isolated_trades.empty else 0
    
    return cluster_analysis


def create_trade_summary_report(trades_df: pd.DataFrame,
                              signals_df: pd.DataFrame) -> str:
    """
    Create a formatted summary report of trading activity.
    
    Args:
        trades_df: DataFrame of trades
        signals_df: Original signals data
        
    Returns:
        Formatted text report
    """
    if trades_df.empty:
        return "No trades found in the data."
    
    analyzer = TradeAnalyzer([])  # Use for utility methods
    stats = analyzer.summary_stats() if not trades_df.empty else {}
    
    report = f"""
Trading Activity Summary
========================

Signal Statistics:
-----------------
Total Signals: {len(signals_df):,}
Unique Strategies: {signals_df['strategy_id'].nunique()}
Bar Range: {signals_df['bar_idx'].min()} - {signals_df['bar_idx'].max()}

Trade Statistics:
----------------
Total Trades: {len(trades_df):,}
Win Rate: {stats.get('win_rate', 0):.1%}
Average Return: {stats.get('avg_return', 0):.2%}
Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}
Profit Factor: {stats.get('profit_factor', 0):.2f}

Best Trade: {stats.get('best_trade', 0):.2%}
Worst Trade: {stats.get('worst_trade', 0):.2%}
Average Duration: {stats.get('avg_duration', 0):.1f} bars

Risk Metrics:
------------
Max Drawdown: {stats.get('max_drawdown', 0):.2%}
Average Winner: {stats.get('avg_winner', 0):.2%}
Average Loser: {stats.get('avg_loser', 0):.2%}

Distribution:
------------
Long Trades: {len(trades_df[trades_df['direction'] == 'long']):,} ({len(trades_df[trades_df['direction'] == 'long'])/len(trades_df):.1%})
Short Trades: {len(trades_df[trades_df['direction'] == 'short']):,} ({len(trades_df[trades_df['direction'] == 'short'])/len(trades_df):.1%})
"""
    
    return report