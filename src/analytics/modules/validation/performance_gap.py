"""
Analyze performance gap between expected and actual execution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_performance_gap(
    expected_trades: pd.DataFrame,
    actual_trades: pd.DataFrame,
    group_by: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate performance differences between expected and actual trades.
    
    Args:
        expected_trades: DataFrame of expected trades
        actual_trades: DataFrame of actual trades
        group_by: Optional columns to group by (e.g., ['strategy_id'])
        
    Returns:
        DataFrame with performance gap analysis
    """
    if expected_trades.empty or actual_trades.empty:
        return pd.DataFrame()
    
    # Default grouping
    if group_by is None:
        group_by = ['strategy_id'] if 'strategy_id' in expected_trades.columns else []
    
    # Calculate metrics for expected trades
    if group_by:
        expected_stats = expected_trades.groupby(group_by).agg({
            'pnl': ['sum', 'mean', 'std', 'count'],
            'duration_bars': 'mean'
        })
    else:
        expected_stats = pd.DataFrame({
            ('pnl', 'sum'): [expected_trades['pnl'].sum()],
            ('pnl', 'mean'): [expected_trades['pnl'].mean()],
            ('pnl', 'std'): [expected_trades['pnl'].std()],
            ('pnl', 'count'): [len(expected_trades)],
            ('duration_bars', 'mean'): [expected_trades['duration_bars'].mean()]
        })
    
    # Calculate metrics for actual trades
    if group_by:
        actual_stats = actual_trades.groupby(group_by).agg({
            'pnl': ['sum', 'mean', 'std', 'count'],
            'duration_bars': 'mean',
            'slippage_entry': 'mean',
            'slippage_exit': 'mean',
            'commission': 'sum'
        })
    else:
        actual_stats = pd.DataFrame({
            ('pnl', 'sum'): [actual_trades['pnl'].sum()],
            ('pnl', 'mean'): [actual_trades['pnl'].mean()],
            ('pnl', 'std'): [actual_trades['pnl'].std()],
            ('pnl', 'count'): [len(actual_trades)],
            ('duration_bars', 'mean'): [actual_trades['duration_bars'].mean()],
            ('slippage_entry', 'mean'): [actual_trades['slippage_entry'].mean()],
            ('slippage_exit', 'mean'): [actual_trades['slippage_exit'].mean()],
            ('commission', 'sum'): [actual_trades['commission'].sum()]
        })
    
    # Flatten column names
    expected_stats.columns = ['_'.join(col) if col[1] else col[0] for col in expected_stats.columns]
    actual_stats.columns = ['_'.join(col) if col[1] else col[0] for col in actual_stats.columns]
    
    # Create comparison dataframe
    comparison = pd.DataFrame(index=expected_stats.index)
    
    # Trade counts
    comparison['expected_trades'] = expected_stats['pnl_count']
    comparison['actual_trades'] = actual_stats['pnl_count']
    comparison['trade_diff'] = actual_stats['pnl_count'] - expected_stats['pnl_count']
    comparison['trade_ratio'] = actual_stats['pnl_count'] / expected_stats['pnl_count']
    
    # PnL comparison
    comparison['expected_pnl'] = expected_stats['pnl_sum']
    comparison['actual_pnl'] = actual_stats['pnl_sum']
    comparison['pnl_gap'] = actual_stats['pnl_sum'] - expected_stats['pnl_sum']
    comparison['pnl_ratio'] = actual_stats['pnl_sum'] / expected_stats['pnl_sum']
    
    # Average trade comparison
    comparison['expected_avg_pnl'] = expected_stats['pnl_mean']
    comparison['actual_avg_pnl'] = actual_stats['pnl_mean']
    comparison['avg_pnl_gap'] = actual_stats['pnl_mean'] - expected_stats['pnl_mean']
    
    # Execution costs
    comparison['total_slippage'] = (actual_stats['slippage_entry_mean'] + 
                                   actual_stats['slippage_exit_mean']) * actual_stats['pnl_count']
    comparison['total_commission'] = actual_stats['commission_sum']
    comparison['total_execution_cost'] = comparison['total_slippage'] + comparison['total_commission']
    
    # Performance attribution
    comparison['strategy_pnl'] = expected_stats['pnl_sum']
    comparison['execution_impact'] = comparison['pnl_gap']
    comparison['execution_pct'] = (comparison['execution_impact'] / 
                                  comparison['strategy_pnl'] * 100).round(1)
    
    return comparison


def analyze_slippage_impact(
    trades_df: pd.DataFrame,
    price_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Analyze the impact of slippage on performance.
    
    Args:
        trades_df: DataFrame with actual trades
        price_data: Optional price data for additional analysis
        
    Returns:
        Dictionary with slippage analysis
    """
    results = {
        'summary': {},
        'by_trade_size': {},
        'by_time_of_day': {},
        'by_market_condition': {}
    }
    
    if trades_df.empty:
        return results
    
    # Summary statistics
    total_trades = len(trades_df)
    avg_entry_slippage = trades_df['slippage_entry'].mean()
    avg_exit_slippage = trades_df['slippage_exit'].mean()
    total_slippage_cost = ((trades_df['slippage_entry'] + trades_df['slippage_exit']) * 100).sum()  # Assume 100 shares
    
    results['summary'] = {
        'total_trades': total_trades,
        'avg_entry_slippage': avg_entry_slippage,
        'avg_exit_slippage': avg_exit_slippage,
        'total_slippage': avg_entry_slippage + avg_exit_slippage,
        'total_slippage_cost': total_slippage_cost,
        'slippage_per_trade': total_slippage_cost / total_trades if total_trades > 0 else 0
    }
    
    # Analyze slippage by trade characteristics
    if 'trade_size' in trades_df.columns:
        size_analysis = trades_df.groupby(pd.qcut(trades_df['trade_size'], q=4)).agg({
            'slippage_entry': 'mean',
            'slippage_exit': 'mean'
        })
        results['by_trade_size'] = size_analysis.to_dict()
    
    # Analyze by time of day if we have timestamps
    if 'entry_fill_time' in trades_df.columns:
        try:
            trades_df['hour'] = pd.to_datetime(trades_df['entry_fill_time']).dt.hour
            hour_analysis = trades_df.groupby('hour').agg({
                'slippage_entry': 'mean',
                'slippage_exit': 'mean',
                'trade_id': 'count'
            })
            results['by_time_of_day'] = hour_analysis.to_dict()
        except:
            pass
    
    # High slippage trades
    high_slippage_threshold = (avg_entry_slippage + avg_exit_slippage) * 2
    high_slippage_trades = trades_df[
        (trades_df['slippage_entry'] + trades_df['slippage_exit']) > high_slippage_threshold
    ]
    
    results['high_slippage_trades'] = {
        'count': len(high_slippage_trades),
        'pct_of_trades': len(high_slippage_trades) / total_trades * 100 if total_trades > 0 else 0,
        'avg_pnl': high_slippage_trades['pnl'].mean() if not high_slippage_trades.empty else 0
    }
    
    return results


def plot_performance_attribution(
    performance_gap_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize performance attribution between strategy and execution.
    
    Args:
        performance_gap_df: Output from calculate_performance_gap
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Performance Attribution Analysis', fontsize=16)
    
    # 1. Expected vs Actual PnL
    ax1 = axes[0, 0]
    if not performance_gap_df.empty:
        x = range(len(performance_gap_df))
        width = 0.35
        ax1.bar([i - width/2 for i in x], performance_gap_df['expected_pnl'], 
                width, label='Expected', alpha=0.8)
        ax1.bar([i + width/2 for i in x], performance_gap_df['actual_pnl'], 
                width, label='Actual', alpha=0.8)
        ax1.set_title('Expected vs Actual PnL')
        ax1.set_ylabel('PnL ($)')
        ax1.legend()
        if len(performance_gap_df) <= 10:
            ax1.set_xticks(x)
            ax1.set_xticklabels(performance_gap_df.index, rotation=45)
    
    # 2. PnL Gap Breakdown
    ax2 = axes[0, 1]
    if not performance_gap_df.empty:
        gap_data = performance_gap_df['pnl_gap'].sort_values()
        colors = ['red' if x < 0 else 'green' for x in gap_data.values]
        ax2.barh(range(len(gap_data)), gap_data.values, color=colors)
        ax2.set_title('PnL Gap by Strategy')
        ax2.set_xlabel('PnL Difference ($)')
        if len(gap_data) <= 10:
            ax2.set_yticks(range(len(gap_data)))
            ax2.set_yticklabels(gap_data.index)
    
    # 3. Execution Cost Components
    ax3 = axes[1, 0]
    if not performance_gap_df.empty:
        cost_data = performance_gap_df[['total_slippage', 'total_commission']].sum()
        ax3.pie(cost_data.values, labels=cost_data.index, autopct='%1.1f%%')
        ax3.set_title('Execution Cost Breakdown')
    
    # 4. Performance Ratio Distribution
    ax4 = axes[1, 1]
    if not performance_gap_df.empty and 'pnl_ratio' in performance_gap_df.columns:
        ratios = performance_gap_df['pnl_ratio'].dropna()
        if not ratios.empty:
            ax4.hist(ratios, bins=20, alpha=0.7, edgecolor='black')
            ax4.axvline(x=1.0, color='red', linestyle='--', label='Perfect Execution')
            ax4.set_title('Performance Ratio Distribution')
            ax4.set_xlabel('Actual/Expected PnL Ratio')
            ax4.set_ylabel('Count')
            ax4.legend()
    
    plt.tight_layout()
    return fig


def create_execution_report_card(
    performance_gap: pd.DataFrame,
    slippage_analysis: Dict[str, Any]
) -> str:
    """
    Create a summary report card for execution quality.
    
    Args:
        performance_gap: Output from calculate_performance_gap
        slippage_analysis: Output from analyze_slippage_impact
        
    Returns:
        Formatted report string
    """
    report = "# Execution Quality Report Card\n\n"
    
    # Overall grade calculation
    if not performance_gap.empty:
        avg_pnl_ratio = performance_gap['pnl_ratio'].mean()
        if avg_pnl_ratio >= 0.95:
            grade = 'A'
        elif avg_pnl_ratio >= 0.90:
            grade = 'B'
        elif avg_pnl_ratio >= 0.85:
            grade = 'C'
        elif avg_pnl_ratio >= 0.80:
            grade = 'D'
        else:
            grade = 'F'
        
        report += f"## Overall Grade: {grade}\n\n"
        report += f"**Performance Ratio**: {avg_pnl_ratio:.2f}\n\n"
    
    # Key metrics
    report += "## Key Metrics\n\n"
    
    if not performance_gap.empty:
        total_expected = performance_gap['expected_pnl'].sum()
        total_actual = performance_gap['actual_pnl'].sum()
        total_gap = performance_gap['pnl_gap'].sum()
        
        report += f"- **Expected PnL**: ${total_expected:,.2f}\n"
        report += f"- **Actual PnL**: ${total_actual:,.2f}\n"
        report += f"- **Performance Gap**: ${total_gap:,.2f} ({total_gap/total_expected*100:.1f}%)\n\n"
    
    # Slippage metrics
    if slippage_analysis and 'summary' in slippage_analysis:
        summary = slippage_analysis['summary']
        report += "## Slippage Analysis\n\n"
        report += f"- **Avg Entry Slippage**: ${summary.get('avg_entry_slippage', 0):.4f}\n"
        report += f"- **Avg Exit Slippage**: ${summary.get('avg_exit_slippage', 0):.4f}\n"
        report += f"- **Total Slippage Cost**: ${summary.get('total_slippage_cost', 0):,.2f}\n"
        report += f"- **Slippage per Trade**: ${summary.get('slippage_per_trade', 0):.2f}\n\n"
    
    # Recommendations
    report += "## Recommendations\n\n"
    
    if not performance_gap.empty:
        if avg_pnl_ratio < 0.90:
            report += "- ⚠️ Significant execution degradation detected\n"
            report += "- Consider reviewing order types and execution timing\n"
        
        high_cost_strategies = performance_gap[performance_gap['execution_pct'] < -10]
        if not high_cost_strategies.empty:
            report += f"- {len(high_cost_strategies)} strategies have >10% execution cost\n"
    
    if slippage_analysis and 'high_slippage_trades' in slippage_analysis:
        high_slip = slippage_analysis['high_slippage_trades']
        if high_slip.get('pct_of_trades', 0) > 5:
            report += f"- {high_slip['pct_of_trades']:.1f}% of trades have high slippage\n"
    
    return report