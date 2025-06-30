"""Trade duration analysis."""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt


def analyze_trade_duration(
    trades_df: pd.DataFrame,
    group_by: Optional[List[str]] = None,
    bins: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Analyze trade holding period durations.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades (must have duration columns)
    group_by : list, optional
        Columns to group by
    bins : list, optional
        Duration bins in minutes (e.g., [0, 5, 15, 60, 240, np.inf])
        
    Returns
    -------
    pd.DataFrame
        Duration analysis results
    """
    if trades_df.empty:
        return pd.DataFrame()
    
    # Default bins if not provided (in minutes)
    if bins is None:
        bins = [0, 5, 15, 30, 60, 120, 240, 480, np.inf]
    
    # Ensure we have duration data
    if 'duration_minutes' not in trades_df.columns:
        if 'duration_hours' in trades_df.columns:
            trades_df['duration_minutes'] = trades_df['duration_hours'] * 60
        else:
            print("Warning: No duration data found")
            return pd.DataFrame()
    
    # Create duration categories
    trades_df = trades_df.copy()
    bin_labels = []
    for i in range(len(bins) - 1):
        if bins[i+1] == np.inf:
            bin_labels.append(f'{bins[i]}+ min')
        else:
            bin_labels.append(f'{bins[i]}-{bins[i+1]} min')
    
    trades_df['duration_category'] = pd.cut(
        trades_df['duration_minutes'], 
        bins=bins,
        labels=bin_labels
    )
    
    # Calculate statistics
    def calc_duration_stats(group):
        stats = {
            'total_trades': len(group),
            'avg_duration_min': group['duration_minutes'].mean(),
            'median_duration_min': group['duration_minutes'].median(),
            'std_duration_min': group['duration_minutes'].std(),
            'min_duration_min': group['duration_minutes'].min(),
            'max_duration_min': group['duration_minutes'].max(),
            
            # Performance by duration
            'avg_return': group['pct_return'].mean() * 100,
            'win_rate': (group['pct_return'] > 0).mean(),
        }
        
        # Duration category breakdown
        duration_counts = group['duration_category'].value_counts()
        for category in bin_labels:
            stats[f'trades_{category}'] = duration_counts.get(category, 0)
            stats[f'pct_{category}'] = duration_counts.get(category, 0) / len(group) * 100
        
        return pd.Series(stats)
    
    # Apply grouping
    if group_by:
        duration_stats = trades_df.groupby(group_by).apply(calc_duration_stats).reset_index()
    else:
        duration_stats = pd.DataFrame([calc_duration_stats(trades_df)])
    
    return duration_stats


def filter_by_duration(
    trades_df: pd.DataFrame,
    min_minutes: Optional[float] = None,
    max_minutes: Optional[float] = None,
    min_hours: Optional[float] = None,
    max_hours: Optional[float] = None
) -> pd.DataFrame:
    """
    Filter trades by duration.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades
    min_minutes : float, optional
        Minimum duration in minutes
    max_minutes : float, optional
        Maximum duration in minutes
    min_hours : float, optional
        Minimum duration in hours
    max_hours : float, optional
        Maximum duration in hours
        
    Returns
    -------
    pd.DataFrame
        Filtered trades
    """
    df = trades_df.copy()
    
    # Convert hours to minutes if specified
    if min_hours is not None:
        min_minutes = min_hours * 60
    if max_hours is not None:
        max_minutes = max_hours * 60
    
    # Ensure we have duration data
    if 'duration_minutes' not in df.columns:
        if 'duration_hours' in df.columns:
            df['duration_minutes'] = df['duration_hours'] * 60
        else:
            print("Warning: No duration data found")
            return df
    
    # Apply filters
    mask = pd.Series([True] * len(df))
    
    if min_minutes is not None:
        mask &= df['duration_minutes'] >= min_minutes
        
    if max_minutes is not None:
        mask &= df['duration_minutes'] <= max_minutes
    
    filtered = df[mask]
    
    print(f"Filtered from {len(df)} to {len(filtered)} trades")
    print(f"Duration range: {min_minutes or 0:.1f} - {max_minutes or 'inf'} minutes")
    
    return filtered


def calculate_holding_periods(
    trades_df: pd.DataFrame,
    market_hours: Tuple[str, str] = ('09:30', '16:00'),
    timezone: str = 'America/New_York'
) -> pd.DataFrame:
    """
    Calculate actual market hours held for each trade.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades
    market_hours : tuple
        Market open and close times
    timezone : str
        Market timezone
        
    Returns
    -------
    pd.DataFrame
        Trades with market hours held
    """
    df = trades_df.copy()
    
    # Ensure timestamps are datetime
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    # Localize to market timezone
    if df['entry_time'].dt.tz is None:
        df['entry_time'] = df['entry_time'].dt.tz_localize(timezone)
        df['exit_time'] = df['exit_time'].dt.tz_localize(timezone)
    else:
        df['entry_time'] = df['entry_time'].dt.tz_convert(timezone)
        df['exit_time'] = df['exit_time'].dt.tz_convert(timezone)
    
    # Calculate market hours for each trade
    market_open = pd.to_datetime(market_hours[0]).time()
    market_close = pd.to_datetime(market_hours[1]).time()
    
    def calc_market_hours(row):
        entry = row['entry_time']
        exit = row['exit_time']
        
        total_minutes = 0
        current = entry
        
        while current.date() <= exit.date():
            # Get market hours for this day
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                day_open = current.replace(
                    hour=market_open.hour,
                    minute=market_open.minute,
                    second=0,
                    microsecond=0
                )
                day_close = current.replace(
                    hour=market_close.hour,
                    minute=market_close.minute,
                    second=0,
                    microsecond=0
                )
                
                # Calculate overlap
                start = max(current if current.date() == entry.date() else day_open, day_open)
                end = min(exit if exit.date() == current.date() else day_close, day_close)
                
                if start < end:
                    total_minutes += (end - start).total_seconds() / 60
            
            # Move to next day
            current = current.replace(hour=0, minute=0, second=0, microsecond=0) + pd.Timedelta(days=1)
        
        return total_minutes
    
    df['market_hours_held'] = df.apply(calc_market_hours, axis=1)
    df['market_hours_held_hours'] = df['market_hours_held'] / 60
    
    return df


def plot_duration_analysis(
    trades_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create duration analysis plots.
    
    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with trades
    figsize : tuple
        Figure size
    """
    if 'duration_minutes' not in trades_df.columns:
        print("No duration data available for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Duration distribution
    axes[0, 0].hist(trades_df['duration_minutes'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(trades_df['duration_minutes'].mean(), color='red', 
                       linestyle='--', label=f"Mean: {trades_df['duration_minutes'].mean():.1f}")
    axes[0, 0].axvline(trades_df['duration_minutes'].median(), color='green', 
                       linestyle='--', label=f"Median: {trades_df['duration_minutes'].median():.1f}")
    axes[0, 0].set_xlabel('Duration (minutes)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Trade Duration Distribution')
    axes[0, 0].legend()
    
    # 2. Duration vs Returns
    axes[0, 1].scatter(trades_df['duration_minutes'], 
                      trades_df['pct_return'] * 100,
                      alpha=0.5)
    axes[0, 1].set_xlabel('Duration (minutes)')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].set_title('Duration vs Returns')
    axes[0, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # 3. Returns by duration bucket
    duration_buckets = pd.cut(trades_df['duration_minutes'], 
                             bins=[0, 5, 15, 30, 60, 120, np.inf],
                             labels=['0-5', '5-15', '15-30', '30-60', '60-120', '120+'])
    
    returns_by_duration = trades_df.groupby(duration_buckets)['pct_return'].agg(['mean', 'count'])
    returns_by_duration['mean'] *= 100  # Convert to percentage
    
    axes[1, 0].bar(range(len(returns_by_duration)), returns_by_duration['mean'])
    axes[1, 0].set_xticks(range(len(returns_by_duration)))
    axes[1, 0].set_xticklabels(returns_by_duration.index)
    axes[1, 0].set_xlabel('Duration Bucket (minutes)')
    axes[1, 0].set_ylabel('Average Return (%)')
    axes[1, 0].set_title('Average Returns by Duration')
    axes[1, 0].axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # Add trade counts on top of bars
    for i, (_, row) in enumerate(returns_by_duration.iterrows()):
        axes[1, 0].text(i, row['mean'] + 0.1 if row['mean'] > 0 else row['mean'] - 0.1, 
                       f"n={int(row['count'])}", ha='center', va='bottom' if row['mean'] > 0 else 'top')
    
    # 4. Win rate by duration
    win_rate_by_duration = trades_df.groupby(duration_buckets).apply(
        lambda x: (x['pct_return'] > 0).mean() * 100
    )
    
    axes[1, 1].bar(range(len(win_rate_by_duration)), win_rate_by_duration)
    axes[1, 1].set_xticks(range(len(win_rate_by_duration)))
    axes[1, 1].set_xticklabels(win_rate_by_duration.index)
    axes[1, 1].set_xlabel('Duration Bucket (minutes)')
    axes[1, 1].set_ylabel('Win Rate (%)')
    axes[1, 1].set_title('Win Rate by Duration')
    axes[1, 1].axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()