"""Stop loss optimization functionality."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


def optimize_stop_loss(
    trades_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    stop_range: List[float] = [0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0],
    metric: str = 'sharpe'
) -> Dict:
    """
    Optimize stop loss percentage.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades
    signals_df : pd.DataFrame
        DataFrame with price data during trades
    stop_range : list
        Stop loss percentages to test
    metric : str
        Optimization metric
        
    Returns
    -------
    dict
        Optimal stop loss and performance metrics
    """
    results = []
    
    for stop_pct in stop_range:
        returns = backtest_with_stops(trades_df, signals_df, stop_pct)
        
        if len(returns) > 0:
            result = {
                'stop': stop_pct,
                'avg_return': np.mean(returns),
                'total_return': (1 + pd.Series(returns)).prod() - 1,
                'sharpe': _calculate_sharpe(returns),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                'max_drawdown': _calculate_max_drawdown(returns),
                'trades_stopped': sum(1 for r in returns if r == -stop_pct/100)
            }
            results.append(result)
    
    if not results:
        return {}
    
    results_df = pd.DataFrame(results)
    
    # Find optimal
    if metric == 'sharpe':
        optimal_idx = results_df['sharpe'].idxmax()
    elif metric == 'return':
        optimal_idx = results_df['total_return'].idxmax()
    else:
        optimal_idx = results_df['win_rate'].idxmax()
    
    optimal = results_df.iloc[optimal_idx].to_dict()
    optimal['all_results'] = results_df
    
    return optimal


def backtest_with_stops(
    trades_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    stop_pct: float,
    target_pct: Optional[float] = None
) -> List[float]:
    """
    Backtest trades with stop loss (and optional profit target).
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades
    signals_df : pd.DataFrame
        DataFrame with price data
    stop_pct : float
        Stop loss percentage
    target_pct : float, optional
        Profit target percentage
        
    Returns
    -------
    list
        Modified returns after applying stops
    """
    returns = []
    
    for _, trade in trades_df.iterrows():
        # Get price data during trade
        mask = (signals_df['timestamp'] >= trade['entry_time']) & \
               (signals_df['timestamp'] <= trade['exit_time'])
        
        if 'symbol' in trade and 'symbol' in signals_df.columns:
            mask &= signals_df['symbol'] == trade['symbol']
        
        trade_prices = signals_df[mask]
        
        if trade_prices.empty:
            returns.append(trade['pct_return'])
            continue
        
        # Apply stops
        final_return = _apply_stops_to_trade(
            trade, trade_prices, stop_pct, target_pct
        )
        
        returns.append(final_return)
    
    return returns


def analyze_stop_effectiveness(
    original_trades: pd.DataFrame,
    stopped_returns: List[float],
    stop_pct: float
) -> Dict:
    """
    Analyze how effective a stop loss was.
    
    Parameters
    ----------
    original_trades : pd.DataFrame
        Original trades without stops
    stopped_returns : list
        Returns after applying stops
    stop_pct : float
        Stop loss percentage used
        
    Returns
    -------
    dict
        Analysis of stop effectiveness
    """
    original_returns = original_trades['pct_return'].tolist()
    
    # Count stopped trades
    n_stopped = sum(1 for r in stopped_returns if r == -stop_pct/100)
    
    # Analyze saved vs lost
    saved_losses = 0
    prevented_gains = 0
    
    for orig, stopped in zip(original_returns, stopped_returns):
        if stopped == -stop_pct/100:
            if orig < stopped:
                # Stop saved us from larger loss
                saved_losses += stopped - orig
            elif orig > stopped:
                # Stop prevented gain
                prevented_gains += orig - stopped
    
    return {
        'stop_pct': stop_pct,
        'trades_stopped': n_stopped,
        'stop_rate': n_stopped / len(original_trades),
        'saved_losses': saved_losses,
        'prevented_gains': prevented_gains,
        'net_benefit': saved_losses - prevented_gains,
        'original_sharpe': _calculate_sharpe(original_returns),
        'stopped_sharpe': _calculate_sharpe(stopped_returns),
        'original_return': (1 + pd.Series(original_returns)).prod() - 1,
        'stopped_return': (1 + pd.Series(stopped_returns)).prod() - 1
    }


# Helper functions
def _apply_stops_to_trade(trade, trade_prices, stop_pct, target_pct=None):
    """Apply stop loss and profit target to a single trade."""
    entry_price = trade['entry_price']
    direction = 1 if trade['direction'] == 'long' else -1
    
    if direction == 1:  # Long
        stop_price = entry_price * (1 - stop_pct/100)
        target_price = entry_price * (1 + target_pct/100) if target_pct else None
        
        for _, bar in trade_prices.iterrows():
            # Check stop
            if 'low' in bar and bar['low'] <= stop_price:
                return -stop_pct/100
            
            # Check target
            if target_price and 'high' in bar and bar['high'] >= target_price:
                return target_pct/100
    
    else:  # Short
        stop_price = entry_price * (1 + stop_pct/100)
        target_price = entry_price * (1 - target_pct/100) if target_pct else None
        
        for _, bar in trade_prices.iterrows():
            # Check stop
            if 'high' in bar and bar['high'] >= stop_price:
                return -stop_pct/100
            
            # Check target
            if target_price and 'low' in bar and bar['low'] <= target_price:
                return target_pct/100
    
    # If no stop/target hit, return original
    return trade['pct_return']


def _calculate_sharpe(returns, periods_per_year=252):
    """Calculate Sharpe ratio."""
    if len(returns) < 2:
        return 0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0
    
    return mean_return / std_return * np.sqrt(periods_per_year)


def _calculate_max_drawdown(returns):
    """Calculate maximum drawdown."""
    cum_returns = (1 + pd.Series(returns)).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()