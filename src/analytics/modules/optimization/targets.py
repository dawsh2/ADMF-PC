"""Profit target optimization functionality."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def optimize_profit_target(
    trades_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    target_range: List[float] = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0],
    stop_loss: Optional[float] = None,
    metric: str = 'sharpe'
) -> Dict:
    """
    Optimize profit target percentage.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades
    signals_df : pd.DataFrame
        DataFrame with price data
    target_range : list
        Profit target percentages to test
    stop_loss : float, optional
        Fixed stop loss to use
    metric : str
        Optimization metric
        
    Returns
    -------
    dict
        Optimal profit target and metrics
    """
    from .stop_loss import backtest_with_stops
    
    results = []
    
    for target_pct in target_range:
        returns = backtest_with_stops(
            trades_df, signals_df, 
            stop_pct=stop_loss or 100,  # Large stop if not specified
            target_pct=target_pct
        )
        
        if len(returns) > 0:
            result = {
                'target': target_pct,
                'avg_return': np.mean(returns),
                'total_return': (1 + pd.Series(returns)).prod() - 1,
                'sharpe': _calculate_sharpe(returns),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                'targets_hit': sum(1 for r in returns if r == target_pct/100)
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


def optimize_stop_target_grid(
    trades_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    stop_range: List[float] = [0.5, 1.0, 2.0],
    target_range: List[float] = [1.0, 2.0, 3.0],
    metric: str = 'sharpe',
    plot: bool = True
) -> Dict:
    """
    Grid search optimization for stop loss and profit target.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades
    signals_df : pd.DataFrame
        DataFrame with price data
    stop_range : list
        Stop loss percentages
    target_range : list
        Profit target percentages
    metric : str
        Optimization metric
    plot : bool
        Whether to plot heatmap
        
    Returns
    -------
    dict
        Optimal parameters and grid results
    """
    from .stop_loss import backtest_with_stops
    
    # Grid search
    results = []
    
    for stop_pct in stop_range:
        for target_pct in target_range:
            returns = backtest_with_stops(
                trades_df, signals_df, stop_pct, target_pct
            )
            
            if len(returns) > 0:
                result = {
                    'stop': stop_pct,
                    'target': target_pct,
                    'sharpe': _calculate_sharpe(returns),
                    'total_return': (1 + pd.Series(returns)).prod() - 1,
                    'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                    'avg_return': np.mean(returns)
                }
                results.append(result)
    
    if not results:
        return {}
    
    results_df = pd.DataFrame(results)
    
    # Create pivot table for heatmap
    pivot = results_df.pivot(index='stop', columns='target', values=metric)
    
    # Find optimal
    optimal_idx = results_df[metric].idxmax()
    optimal = results_df.iloc[optimal_idx].to_dict()
    
    if plot and len(pivot) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0)
        plt.title(f'{metric.capitalize()} by Stop/Target Grid')
        plt.xlabel('Profit Target %')
        plt.ylabel('Stop Loss %')
        
        # Mark optimal
        opt_stop_idx = list(pivot.index).index(optimal['stop'])
        opt_target_idx = list(pivot.columns).index(optimal['target'])
        plt.scatter(opt_target_idx + 0.5, opt_stop_idx + 0.5, 
                   marker='*', s=500, c='blue', edgecolors='black', linewidth=2)
        
        plt.tight_layout()
        plt.show()
    
    optimal['grid_results'] = results_df
    optimal['pivot_table'] = pivot
    
    return optimal


def calculate_risk_reward_ratio(
    stop_loss: float,
    profit_target: float
) -> float:
    """
    Calculate risk/reward ratio.
    
    Parameters
    ----------
    stop_loss : float
        Stop loss percentage
    profit_target : float
        Profit target percentage
        
    Returns
    -------
    float
        Risk/reward ratio
    """
    return profit_target / stop_loss


def analyze_target_efficiency(
    trades_df: pd.DataFrame,
    target_results: Dict
) -> pd.DataFrame:
    """
    Analyze how efficiently targets are being hit.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        Original trades
    target_results : dict
        Results from optimize_profit_target
        
    Returns
    -------
    pd.DataFrame
        Target efficiency analysis
    """
    if 'all_results' not in target_results:
        return pd.DataFrame()
    
    results = target_results['all_results'].copy()
    
    # Calculate efficiency metrics
    results['target_efficiency'] = results['targets_hit'] / len(trades_df)
    results['capture_ratio'] = results['avg_return'] / (results['target'] / 100)
    
    # Risk/reward if stop loss was used
    if 'stop' in target_results:
        results['risk_reward'] = results['target'] / target_results['stop']
    
    return results


# Helper functions
def _calculate_sharpe(returns, periods_per_year=252):
    """Calculate Sharpe ratio."""
    if len(returns) < 2:
        return 0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0
    
    return mean_return / std_return * np.sqrt(periods_per_year)