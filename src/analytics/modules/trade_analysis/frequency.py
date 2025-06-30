"""Trade frequency analysis and filtering."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt


def calculate_trade_frequency(
    trades_df: pd.DataFrame,
    group_by: Optional[List[str]] = None,
    period: str = 'day'
) -> pd.DataFrame:
    """
    Calculate trade frequency statistics.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades (must have entry_time and exit_time)
    group_by : list, optional
        Columns to group by (e.g., ['symbol', 'strategy'])
    period : str
        Period for frequency calculation ('day', 'week', 'month')
        
    Returns
    -------
    pd.DataFrame
        Trade frequency statistics
        
    Examples
    --------
    >>> freq_stats = calculate_trade_frequency(trades, group_by=['strategy'])
    >>> print(freq_stats[['strategy', 'trades_per_day', 'avg_duration_hours']])
    """
    if trades_df.empty:
        return pd.DataFrame()
    
    if group_by is None:
        group_by = []
        # Auto-detect grouping columns
        for col in ['symbol', 'strategy', 'strategy_hash']:
            if col in trades_df.columns:
                group_by.append(col)
    
    # Convert timestamps
    trades_df = trades_df.copy()
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    # Calculate base statistics
    def calc_stats(group):
        # Date range
        start_date = group['entry_time'].min()
        end_date = group['exit_time'].max()
        
        # Calculate period count
        if period == 'day':
            period_count = (end_date - start_date).days + 1
        elif period == 'week':
            period_count = ((end_date - start_date).days + 1) / 7
        elif period == 'month':
            period_count = ((end_date - start_date).days + 1) / 30
        else:
            period_count = 1
        
        # Basic stats
        stats = {
            'total_trades': len(group),
            'start_date': start_date,
            'end_date': end_date,
            f'{period}s_active': period_count,
            f'trades_per_{period}': len(group) / max(period_count, 1),
            
            # Duration stats
            'avg_duration_minutes': group['duration_minutes'].mean() if 'duration_minutes' in group else np.nan,
            'avg_duration_hours': group['duration_hours'].mean() if 'duration_hours' in group else np.nan,
            'min_duration_minutes': group['duration_minutes'].min() if 'duration_minutes' in group else np.nan,
            'max_duration_minutes': group['duration_minutes'].max() if 'duration_minutes' in group else np.nan,
            
            # Return stats
            'avg_return': group['pct_return'].mean(),
            'total_return': (1 + group['pct_return']).prod() - 1,
            'win_rate': (group['pct_return'] > 0).mean(),
            
            # Direction stats
            'long_trades': (group['direction'] == 'long').sum() if 'direction' in group else np.nan,
            'short_trades': (group['direction'] == 'short').sum() if 'direction' in group else np.nan,
        }
        
        return pd.Series(stats)
    
    # Apply grouping
    if group_by:
        freq_stats = trades_df.groupby(group_by).apply(calc_stats).reset_index()
    else:
        freq_stats = pd.DataFrame([calc_stats(trades_df)])
    
    # Sort by trade frequency
    freq_stats = freq_stats.sort_values(f'trades_per_{period}', ascending=False)
    
    return freq_stats


def filter_by_frequency(
    trades_df: pd.DataFrame,
    min_per_day: Optional[float] = None,
    max_per_day: Optional[float] = None,
    group_by: Optional[List[str]] = None,
    return_stats: bool = False
) -> pd.DataFrame:
    """
    Filter trades by frequency criteria.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades
    min_per_day : float, optional
        Minimum trades per day
    max_per_day : float, optional
        Maximum trades per day
    group_by : list, optional
        Columns to group by
    return_stats : bool
        If True, return (filtered_trades, frequency_stats)
        
    Returns
    -------
    pd.DataFrame or tuple
        Filtered trades (and optionally frequency stats)
        
    Examples
    --------
    >>> # Keep only strategies trading 2-10 times per day
    >>> active_trades = filter_by_frequency(trades, min_per_day=2, max_per_day=10)
    
    >>> # Get both filtered trades and stats
    >>> trades, stats = filter_by_frequency(trades, min_per_day=5, return_stats=True)
    """
    # Calculate frequency stats
    freq_stats = calculate_trade_frequency(trades_df, group_by=group_by, period='day')
    
    # Apply filters
    mask = pd.Series([True] * len(freq_stats))
    
    if min_per_day is not None:
        mask &= freq_stats['trades_per_day'] >= min_per_day
        
    if max_per_day is not None:
        mask &= freq_stats['trades_per_day'] <= max_per_day
    
    qualified_stats = freq_stats[mask]
    
    # Filter trades
    if not group_by:
        # If no grouping, apply filter to all trades
        if mask.any():
            filtered_trades = trades_df.copy()
        else:
            filtered_trades = pd.DataFrame()
    else:
        # Filter based on group membership
        filtered_trades = []
        
        for _, row in qualified_stats.iterrows():
            # Build filter condition
            condition = pd.Series([True] * len(trades_df))
            
            for col in group_by:
                if col in trades_df.columns:
                    condition &= trades_df[col] == row[col]
            
            filtered_trades.append(trades_df[condition])
        
        if filtered_trades:
            filtered_trades = pd.concat(filtered_trades, ignore_index=True)
        else:
            filtered_trades = pd.DataFrame()
    
    print(f"Filtered from {len(trades_df)} to {len(filtered_trades)} trades")
    print(f"Kept {len(qualified_stats)} out of {len(freq_stats)} groups")
    
    if return_stats:
        return filtered_trades, qualified_stats
    else:
        return filtered_trades


def analyze_frequency_distribution(
    trades_df: pd.DataFrame,
    group_by: Optional[List[str]] = None,
    bins: int = 20,
    plot: bool = True
) -> pd.DataFrame:
    """
    Analyze distribution of trade frequencies.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades
    group_by : list, optional
        Columns to group by
    bins : int
        Number of bins for histogram
    plot : bool
        Whether to create plots
        
    Returns
    -------
    pd.DataFrame
        Frequency distribution statistics
    """
    freq_stats = calculate_trade_frequency(trades_df, group_by=group_by)
    
    if freq_stats.empty:
        return pd.DataFrame()
    
    # Distribution statistics
    dist_stats = {
        'mean_trades_per_day': freq_stats['trades_per_day'].mean(),
        'median_trades_per_day': freq_stats['trades_per_day'].median(),
        'std_trades_per_day': freq_stats['trades_per_day'].std(),
        'min_trades_per_day': freq_stats['trades_per_day'].min(),
        'max_trades_per_day': freq_stats['trades_per_day'].max(),
        'strategies_count': len(freq_stats),
        
        # Percentiles
        'p10_trades_per_day': freq_stats['trades_per_day'].quantile(0.1),
        'p25_trades_per_day': freq_stats['trades_per_day'].quantile(0.25),
        'p75_trades_per_day': freq_stats['trades_per_day'].quantile(0.75),
        'p90_trades_per_day': freq_stats['trades_per_day'].quantile(0.9),
    }
    
    if plot and len(freq_stats) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Histogram
        axes[0].hist(freq_stats['trades_per_day'], bins=bins, edgecolor='black', alpha=0.7)
        axes[0].axvline(dist_stats['mean_trades_per_day'], color='red', 
                       linestyle='--', label=f"Mean: {dist_stats['mean_trades_per_day']:.1f}")
        axes[0].axvline(dist_stats['median_trades_per_day'], color='green', 
                       linestyle='--', label=f"Median: {dist_stats['median_trades_per_day']:.1f}")
        axes[0].set_xlabel('Trades per Day')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of Trade Frequencies')
        axes[0].legend()
        
        # Scatter: Frequency vs Performance
        if 'avg_return' in freq_stats.columns:
            axes[1].scatter(freq_stats['trades_per_day'], 
                          freq_stats['avg_return'] * 100,
                          alpha=0.6)
            axes[1].set_xlabel('Trades per Day')
            axes[1].set_ylabel('Average Return (%)')
            axes[1].set_title('Trade Frequency vs Performance')
            axes[1].axhline(0, color='black', linestyle='-', alpha=0.3)
            
            # Add trend line
            if len(freq_stats) > 3:
                z = np.polyfit(freq_stats['trades_per_day'], 
                              freq_stats['avg_return'] * 100, 1)
                p = np.poly1d(z)
                axes[1].plot(sorted(freq_stats['trades_per_day']), 
                           p(sorted(freq_stats['trades_per_day'])),
                           "r--", alpha=0.8, label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
                axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    return pd.DataFrame([dist_stats])