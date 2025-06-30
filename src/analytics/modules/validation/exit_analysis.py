"""
Analyze exit reasons and their impact on performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def analyze_exit_reasons(
    trades_df: pd.DataFrame,
    group_by: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze trade exits by reason and calculate statistics.
    
    Args:
        trades_df: DataFrame with trade records
        group_by: Optional list of columns to group by (e.g., ['strategy_id', 'symbol'])
        
    Returns:
        DataFrame with exit reason statistics
    """
    if trades_df.empty:
        return pd.DataFrame()
    
    # Default grouping
    if group_by is None:
        group_by = []
    
    # Always include exit_reason in grouping
    group_cols = group_by + ['exit_reason']
    
    # Calculate statistics by exit reason
    stats = trades_df.groupby(group_cols).agg({
        'trade_id': 'count',
        'pnl': ['sum', 'mean', 'std'],
        'duration_bars': ['mean', 'std'],
        'slippage_entry': 'mean',
        'slippage_exit': 'mean',
        'commission': 'sum'
    }).round(2)
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                     for col in stats.columns.values]
    
    # Rename columns for clarity
    stats = stats.rename(columns={
        'trade_id_count': 'count',
        'pnl_sum': 'total_pnl',
        'pnl_mean': 'avg_pnl',
        'pnl_std': 'pnl_std',
        'duration_bars_mean': 'avg_duration',
        'duration_bars_std': 'duration_std',
        'slippage_entry_mean': 'avg_entry_slippage',
        'slippage_exit_mean': 'avg_exit_slippage',
        'commission_sum': 'total_commission'
    })
    
    # Calculate percentage of trades
    total_trades = len(trades_df)
    stats['pct_of_trades'] = (stats['count'] / total_trades * 100).round(1)
    
    # Calculate win rate by exit reason
    def calculate_win_rate(group):
        if len(group) == 0:
            return 0
        return (group['pnl'] > 0).sum() / len(group) * 100
    
    win_rates = trades_df.groupby(group_cols).apply(calculate_win_rate)
    stats['win_rate'] = win_rates.round(1)
    
    # Calculate profit factor
    def calculate_profit_factor(group):
        profits = group[group['pnl'] > 0]['pnl'].sum()
        losses = abs(group[group['pnl'] < 0]['pnl'].sum())
        if losses == 0:
            return np.inf if profits > 0 else 0
        return profits / losses
    
    profit_factors = trades_df.groupby(group_cols).apply(calculate_profit_factor)
    stats['profit_factor'] = profit_factors.round(2)
    
    # Sort by count descending
    stats = stats.sort_values('count', ascending=False)
    
    return stats


def plot_exit_breakdown(
    trades_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create visualizations of exit reason breakdown.
    
    Args:
        trades_df: DataFrame with trade records
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Trade Exit Analysis', fontsize=16)
    
    # 1. Exit reason distribution (pie chart)
    ax1 = axes[0, 0]
    exit_counts = trades_df['exit_reason'].value_counts()
    ax1.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%')
    ax1.set_title('Exit Reason Distribution')
    
    # 2. PnL by exit reason (box plot)
    ax2 = axes[0, 1]
    exit_reasons = trades_df['exit_reason'].unique()
    pnl_by_reason = [trades_df[trades_df['exit_reason'] == reason]['pnl'].values 
                     for reason in exit_reasons]
    ax2.boxplot(pnl_by_reason, labels=exit_reasons)
    ax2.set_title('PnL Distribution by Exit Reason')
    ax2.set_ylabel('PnL ($)')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Average duration by exit reason
    ax3 = axes[1, 0]
    avg_duration = trades_df.groupby('exit_reason')['duration_bars'].mean().sort_values()
    ax3.barh(avg_duration.index, avg_duration.values)
    ax3.set_title('Average Trade Duration by Exit Reason')
    ax3.set_xlabel('Duration (bars)')
    
    # 4. Win rate by exit reason
    ax4 = axes[1, 1]
    win_rates = trades_df.groupby('exit_reason').apply(
        lambda x: (x['pnl'] > 0).sum() / len(x) * 100
    ).sort_values()
    ax4.barh(win_rates.index, win_rates.values, color='green')
    ax4.set_title('Win Rate by Exit Reason')
    ax4.set_xlabel('Win Rate (%)')
    ax4.axvline(x=50, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def analyze_premature_exits(
    trades_df: pd.DataFrame,
    expected_trades_df: Optional[pd.DataFrame] = None,
    threshold_bars: int = 5
) -> pd.DataFrame:
    """
    Find trades that exited earlier than expected.
    
    Args:
        trades_df: Actual trades DataFrame
        expected_trades_df: Expected trades DataFrame (optional)
        threshold_bars: Minimum duration to not be considered premature
        
    Returns:
        DataFrame of potentially premature exits
    """
    # Find short duration trades with risk-based exits
    risk_exits = ['stop_loss', 'max_duration', 'regime_change']
    
    premature = trades_df[
        (trades_df['exit_reason'].isin(risk_exits)) &
        (trades_df['duration_bars'] < threshold_bars)
    ].copy()
    
    if premature.empty:
        return pd.DataFrame()
    
    # Add analysis columns
    premature['exit_type'] = 'premature_risk'
    
    # If we have expected trades, compare durations
    if expected_trades_df is not None and not expected_trades_df.empty:
        # Match trades and calculate duration difference
        # This would require more sophisticated matching logic
        pass
    
    # Sort by PnL impact
    premature = premature.sort_values('pnl', ascending=True)
    
    return premature


def calculate_exit_optimization_potential(
    trades_df: pd.DataFrame,
    signal_data: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Calculate potential improvement from optimizing exits.
    
    Args:
        trades_df: DataFrame with actual trades
        signal_data: Optional signal data to check for better exits
        
    Returns:
        Dictionary with optimization metrics
    """
    results = {
        'current_total_pnl': trades_df['pnl'].sum(),
        'stop_loss_impact': 0,
        'premature_exit_impact': 0,
        'late_exit_impact': 0,
        'total_optimization_potential': 0
    }
    
    # Calculate impact of stop losses
    stop_loss_trades = trades_df[trades_df['exit_reason'] == 'stop_loss']
    if not stop_loss_trades.empty:
        results['stop_loss_impact'] = stop_loss_trades['pnl'].sum()
        results['stop_loss_count'] = len(stop_loss_trades)
        results['avg_stop_loss'] = stop_loss_trades['pnl'].mean()
    
    # Calculate impact of other risk exits
    risk_exits = trades_df[trades_df['exit_reason'].isin(['max_duration', 'regime_change', 'eod'])]
    if not risk_exits.empty:
        # These might have exited too early
        results['premature_exit_impact'] = risk_exits[risk_exits['pnl'] < 0]['pnl'].sum()
    
    # If we have signal data, check for better exit points
    if signal_data is not None:
        # This would require sophisticated analysis of alternative exit points
        pass
    
    # Calculate total optimization potential
    results['total_optimization_potential'] = abs(results['stop_loss_impact']) + \
                                            abs(results['premature_exit_impact'])
    
    return results