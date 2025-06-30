"""Regime-specific optimization functionality."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def optimize_stops_by_regime(
    trades_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    regime_column: str,
    stop_range: List[float] = [0.1, 0.2, 0.3, 0.5, 1.0],
    target_range: List[float] = [0.2, 0.5, 1.0, 2.0],
    metric: str = 'sharpe',
    min_trades: int = 10
) -> Dict[str, Dict]:
    """
    Optimize stop loss and profit target for each regime.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades
    signals_df : pd.DataFrame
        DataFrame with signals and regime data
    regime_column : str
        Name of regime column
    stop_range : list
        Stop loss percentages to test
    target_range : list
        Profit target percentages to test
    metric : str
        Optimization metric ('sharpe', 'return', 'win_rate')
    min_trades : int
        Minimum trades required per regime
        
    Returns
    -------
    dict
        Optimal parameters for each regime
        
    Examples
    --------
    >>> optimal = optimize_stops_by_regime(
    ...     trades, signals, 'volatility_regime',
    ...     stop_range=[0.5, 1.0, 2.0],
    ...     target_range=[1.0, 2.0, 3.0]
    ... )
    >>> print(optimal['high_vol'])
    {'stop': 2.0, 'target': 3.0, 'sharpe': 1.5}
    """
    # First, assign regimes to trades
    trades_with_regime = _assign_regimes_to_trades(trades_df, signals_df, regime_column)
    
    # Get unique regimes
    regimes = trades_with_regime[regime_column].dropna().unique()
    
    optimal_params = {}
    
    for regime in regimes:
        regime_trades = trades_with_regime[trades_with_regime[regime_column] == regime]
        
        if len(regime_trades) < min_trades:
            print(f"Skipping {regime}: only {len(regime_trades)} trades (min: {min_trades})")
            continue
        
        print(f"\nOptimizing for {regime} ({len(regime_trades)} trades)...")
        
        # Grid search
        results = []
        
        for stop_pct in stop_range:
            for target_pct in target_range:
                # Backtest with these parameters
                returns = _backtest_regime_stops(
                    regime_trades, signals_df, stop_pct, target_pct
                )
                
                if len(returns) > 0:
                    # Calculate metrics
                    result = {
                        'stop': stop_pct,
                        'target': target_pct,
                        'avg_return': np.mean(returns),
                        'total_return': (1 + pd.Series(returns)).prod() - 1,
                        'sharpe': _calculate_sharpe(returns),
                        'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                        'max_drawdown': _calculate_max_drawdown(returns),
                        'trades': len(returns)
                    }
                    results.append(result)
        
        if results:
            results_df = pd.DataFrame(results)
            
            # Find optimal based on metric
            if metric == 'sharpe':
                optimal_idx = results_df['sharpe'].idxmax()
            elif metric == 'return':
                optimal_idx = results_df['total_return'].idxmax()
            elif metric == 'win_rate':
                optimal_idx = results_df['win_rate'].idxmax()
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            optimal = results_df.iloc[optimal_idx]
            
            optimal_params[regime] = {
                'stop': optimal['stop'],
                'target': optimal['target'],
                metric: optimal[metric],
                'avg_return': optimal['avg_return'],
                'win_rate': optimal['win_rate'],
                'trades': optimal['trades']
            }
            
            print(f"Optimal for {regime}: Stop={optimal['stop']}%, "
                  f"Target={optimal['target']}%, {metric}={optimal[metric]:.3f}")
    
    return optimal_params


def get_regime_optimal_params(
    current_regime: str,
    optimal_params: Dict[str, Dict],
    default_stop: float = 1.0,
    default_target: float = 2.0
) -> Tuple[float, float]:
    """
    Get optimal stop/target for current regime.
    
    Parameters
    ----------
    current_regime : str
        Current market regime
    optimal_params : dict
        Optimal parameters by regime (from optimize_stops_by_regime)
    default_stop : float
        Default stop loss if regime not found
    default_target : float
        Default profit target if regime not found
        
    Returns
    -------
    tuple
        (stop_loss, profit_target)
    """
    if current_regime in optimal_params:
        params = optimal_params[current_regime]
        return params['stop'], params['target']
    else:
        return default_stop, default_target


def apply_regime_stops(
    signals_df: pd.DataFrame,
    optimal_params: Dict[str, Dict],
    regime_column: str,
    signal_col: str = 'signal'
) -> pd.DataFrame:
    """
    Apply regime-specific stops to signals.
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame with signals and regime data
    optimal_params : dict
        Optimal parameters by regime
    regime_column : str
        Name of regime column
    signal_col : str
        Name of signal column
        
    Returns
    -------
    pd.DataFrame
        Trades with regime-specific stops applied
    """
    from ..trade_analysis.extraction import extract_trades_with_stops
    
    trades = []
    
    # Process each regime separately
    for regime in signals_df[regime_column].dropna().unique():
        if regime not in optimal_params:
            continue
        
        regime_signals = signals_df[signals_df[regime_column] == regime]
        
        if not regime_signals.empty:
            params = optimal_params[regime]
            
            regime_trades = extract_trades_with_stops(
                regime_signals,
                stop_loss=params['stop'] / 100,
                take_profit=params['target'] / 100,
                signal_col=signal_col
            )
            
            if not regime_trades.empty:
                regime_trades['applied_regime'] = regime
                regime_trades['applied_stop'] = params['stop']
                regime_trades['applied_target'] = params['target']
                trades.append(regime_trades)
    
    if trades:
        return pd.concat(trades, ignore_index=True)
    else:
        return pd.DataFrame()


def plot_regime_optimization_results(
    optimal_params: Dict[str, Dict],
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize regime-specific optimization results.
    
    Parameters
    ----------
    optimal_params : dict
        Optimal parameters by regime
    figsize : tuple
        Figure size
    """
    if not optimal_params:
        print("No optimization results to plot")
        return
    
    # Convert to DataFrame for easier plotting
    results_data = []
    for regime, params in optimal_params.items():
        data = params.copy()
        data['regime'] = regime
        results_data.append(data)
    
    results_df = pd.DataFrame(results_data)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Stop/Target by regime
    ax = axes[0, 0]
    x = np.arange(len(results_df))
    width = 0.35
    
    ax.bar(x - width/2, results_df['stop'], width, label='Stop %', color='coral')
    ax.bar(x + width/2, results_df['target'], width, label='Target %', color='skyblue')
    ax.set_xlabel('Regime')
    ax.set_ylabel('Percentage')
    ax.set_title('Optimal Stop/Target by Regime')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['regime'])
    ax.legend()
    
    # 2. Performance metric by regime
    ax = axes[0, 1]
    metric_col = next((col for col in ['sharpe', 'avg_return', 'win_rate'] if col in results_df), None)
    if metric_col:
        results_df[metric_col].plot(kind='bar', ax=ax, color='green')
        ax.set_xticklabels(results_df['regime'], rotation=45)
        ax.set_ylabel(metric_col.capitalize())
        ax.set_title(f'{metric_col.capitalize()} by Regime')
    
    # 3. Risk/Reward visualization
    ax = axes[1, 0]
    ax.scatter(results_df['stop'], results_df['target'], s=100)
    for i, row in results_df.iterrows():
        ax.annotate(row['regime'], (row['stop'], row['target']), 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Stop Loss %')
    ax.set_ylabel('Profit Target %')
    ax.set_title('Risk/Reward by Regime')
    ax.plot([0, max(results_df['stop'])], [0, max(results_df['stop'])], 
            'k--', alpha=0.3, label='1:1 Risk/Reward')
    ax.legend()
    
    # 4. Trade count by regime
    ax = axes[1, 1]
    if 'trades' in results_df:
        results_df['trades'].plot(kind='bar', ax=ax, color='purple')
        ax.set_xticklabels(results_df['regime'], rotation=45)
        ax.set_ylabel('Number of Trades')
        ax.set_title('Trade Count by Regime')
    
    plt.tight_layout()
    plt.show()


# Helper functions
def _assign_regimes_to_trades(trades_df, signals_df, regime_column):
    """Assign regime to each trade based on entry time."""
    trades_with_regime = trades_df.copy()
    trades_with_regime[regime_column] = None
    
    # Ensure timestamps are datetime
    trades_with_regime['entry_time'] = pd.to_datetime(trades_with_regime['entry_time'])
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    
    for idx, trade in trades_with_regime.iterrows():
        # Find regime at entry
        mask = signals_df['timestamp'] <= trade['entry_time']
        if 'symbol' in trade and 'symbol' in signals_df.columns:
            mask &= signals_df['symbol'] == trade['symbol']
        
        matching = signals_df[mask]
        if not matching.empty and regime_column in matching.columns:
            trades_with_regime.at[idx, regime_column] = matching[regime_column].iloc[-1]
    
    return trades_with_regime


def _backtest_regime_stops(regime_trades, signals_df, stop_pct, target_pct):
    """Backtest trades with specific stop/target."""
    returns = []
    
    for _, trade in regime_trades.iterrows():
        # Get price data during trade
        mask = (signals_df['timestamp'] >= trade['entry_time']) & \
               (signals_df['timestamp'] <= trade['exit_time'])
        
        if 'symbol' in trade and 'symbol' in signals_df.columns:
            mask &= signals_df['symbol'] == trade['symbol']
        
        trade_prices = signals_df[mask]
        
        if trade_prices.empty:
            returns.append(trade['pct_return'])
            continue
        
        # Simulate stop/target exits
        entry_price = trade['entry_price']
        direction = 1 if trade['direction'] == 'long' else -1
        
        if direction == 1:  # Long
            stop_price = entry_price * (1 - stop_pct/100)
            target_price = entry_price * (1 + target_pct/100)
            
            # Check if stop hit
            if (trade_prices['low'] <= stop_price).any():
                returns.append(-stop_pct/100)
            # Check if target hit
            elif (trade_prices['high'] >= target_price).any():
                returns.append(target_pct/100)
            else:
                returns.append(trade['pct_return'])
        else:  # Short
            stop_price = entry_price * (1 + stop_pct/100)
            target_price = entry_price * (1 - target_pct/100)
            
            # Check if stop hit
            if (trade_prices['high'] >= stop_price).any():
                returns.append(-stop_pct/100)
            # Check if target hit
            elif (trade_prices['low'] <= target_price).any():
                returns.append(target_pct/100)
            else:
                returns.append(trade['pct_return'])
    
    return returns


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