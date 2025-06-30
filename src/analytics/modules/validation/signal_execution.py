"""
Validate signal execution by comparing expected vs actual trades.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def validate_execution(
    expected_trades: pd.DataFrame,
    actual_trades: pd.DataFrame,
    tolerance_bars: int = 1
) -> Dict[str, Any]:
    """
    Compare expected trades with actual execution results.
    
    Args:
        expected_trades: DataFrame of expected trades from signals
        actual_trades: DataFrame of actual trades from execution
        tolerance_bars: Allowed bar index difference for matching (default 1)
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'summary': {},
        'matched_trades': [],
        'unmatched_expected': [],
        'unmatched_actual': [],
        'exit_reason_analysis': {},
        'timing_analysis': {},
        'performance_comparison': {}
    }
    
    if expected_trades.empty or actual_trades.empty:
        results['summary'] = {
            'expected_count': len(expected_trades),
            'actual_count': len(actual_trades),
            'match_rate': 0.0
        }
        return results
    
    # Create copies to avoid modifying originals
    expected = expected_trades.copy()
    actual = actual_trades.copy()
    
    # Match trades by strategy_id and entry bar
    matched_pairs = []
    matched_actual_indices = set()
    
    for idx, exp_trade in expected.iterrows():
        # Find matching actual trade
        strategy_match = actual[actual['strategy_id'] == exp_trade['strategy_id']]
        
        # Match by entry bar within tolerance
        bar_match = strategy_match[
            (strategy_match['entry_bar_idx'] >= exp_trade['entry_bar_idx'] - tolerance_bars) &
            (strategy_match['entry_bar_idx'] <= exp_trade['entry_bar_idx'] + tolerance_bars)
        ]
        
        if not bar_match.empty and bar_match.index[0] not in matched_actual_indices:
            # Found a match
            act_idx = bar_match.index[0]
            matched_actual_indices.add(act_idx)
            
            matched_pairs.append({
                'expected_trade_id': exp_trade['trade_id'],
                'actual_trade_id': bar_match.iloc[0]['trade_id'],
                'strategy_id': exp_trade['strategy_id'],
                'entry_bar_diff': bar_match.iloc[0]['entry_bar_idx'] - exp_trade['entry_bar_idx'],
                'exit_bar_diff': bar_match.iloc[0]['exit_bar_idx'] - exp_trade['exit_bar_idx'],
                'exit_reason_match': exp_trade['exit_reason'] == bar_match.iloc[0]['exit_reason'],
                'actual_exit_reason': bar_match.iloc[0]['exit_reason'],
                'pnl_expected': exp_trade['pnl'],
                'pnl_actual': bar_match.iloc[0]['pnl'],
                'pnl_diff': bar_match.iloc[0]['pnl'] - exp_trade['pnl']
            })
            results['matched_trades'].append(matched_pairs[-1])
        else:
            results['unmatched_expected'].append(exp_trade.to_dict())
    
    # Find unmatched actual trades
    unmatched_actual_indices = set(actual.index) - matched_actual_indices
    for idx in unmatched_actual_indices:
        results['unmatched_actual'].append(actual.loc[idx].to_dict())
    
    # Calculate summary statistics
    total_expected = len(expected)
    total_actual = len(actual)
    total_matched = len(matched_pairs)
    
    results['summary'] = {
        'expected_count': total_expected,
        'actual_count': total_actual,
        'matched_count': total_matched,
        'match_rate': total_matched / total_expected if total_expected > 0 else 0.0,
        'unmatched_expected': len(results['unmatched_expected']),
        'unmatched_actual': len(results['unmatched_actual'])
    }
    
    # Exit reason analysis
    if matched_pairs:
        matched_df = pd.DataFrame(matched_pairs)
        exit_reasons = matched_df.groupby('actual_exit_reason').agg({
            'expected_trade_id': 'count',
            'exit_reason_match': 'sum'
        }).rename(columns={'expected_trade_id': 'count'})
        
        results['exit_reason_analysis'] = exit_reasons.to_dict()
    
    # Timing analysis
    if matched_pairs:
        results['timing_analysis'] = {
            'avg_entry_delay': np.mean([m['entry_bar_diff'] for m in matched_pairs]),
            'avg_exit_delay': np.mean([m['exit_bar_diff'] for m in matched_pairs]),
            'entry_delays': [m['entry_bar_diff'] for m in matched_pairs],
            'exit_delays': [m['exit_bar_diff'] for m in matched_pairs]
        }
    
    # Performance comparison
    if matched_pairs:
        expected_pnl = sum(m['pnl_expected'] for m in matched_pairs)
        actual_pnl = sum(m['pnl_actual'] for m in matched_pairs)
        
        results['performance_comparison'] = {
            'total_expected_pnl': expected_pnl,
            'total_actual_pnl': actual_pnl,
            'pnl_difference': actual_pnl - expected_pnl,
            'pnl_ratio': actual_pnl / expected_pnl if expected_pnl != 0 else 0,
            'avg_pnl_diff_per_trade': np.mean([m['pnl_diff'] for m in matched_pairs])
        }
    
    return results


def create_validation_report(
    validation_results: Dict[str, Any],
    output_format: str = 'markdown'
) -> str:
    """
    Create a formatted validation report.
    
    Args:
        validation_results: Results from validate_execution
        output_format: 'markdown' or 'text'
        
    Returns:
        Formatted report string
    """
    summary = validation_results['summary']
    
    if output_format == 'markdown':
        report = "# Trade Execution Validation Report\n\n"
        
        # Summary section
        report += "## Summary\n\n"
        report += f"- Expected Trades: {summary['expected_count']}\n"
        report += f"- Actual Trades: {summary['actual_count']}\n"
        report += f"- Matched Trades: {summary.get('matched_count', 0)}\n"
        report += f"- Match Rate: {summary['match_rate']:.1%}\n"
        report += f"- Unmatched Expected: {summary['unmatched_expected']}\n"
        report += f"- Unmatched Actual: {summary['unmatched_actual']}\n\n"
        
        # Exit reason analysis
        if validation_results['exit_reason_analysis']:
            report += "## Exit Reason Analysis\n\n"
            report += "| Exit Reason | Count | Matched Expected |\n"
            report += "|------------|-------|------------------|\n"
            
            for reason, data in validation_results['exit_reason_analysis'].items():
                if isinstance(data, dict):
                    count = data.get('count', 0)
                    matched = data.get('exit_reason_match', 0)
                    report += f"| {reason} | {count} | {matched} |\n"
            report += "\n"
        
        # Timing analysis
        if validation_results['timing_analysis']:
            timing = validation_results['timing_analysis']
            report += "## Timing Analysis\n\n"
            report += f"- Average Entry Delay: {timing['avg_entry_delay']:.1f} bars\n"
            report += f"- Average Exit Delay: {timing['avg_exit_delay']:.1f} bars\n\n"
        
        # Performance comparison
        if validation_results['performance_comparison']:
            perf = validation_results['performance_comparison']
            report += "## Performance Comparison\n\n"
            report += f"- Expected Total PnL: ${perf['total_expected_pnl']:.2f}\n"
            report += f"- Actual Total PnL: ${perf['total_actual_pnl']:.2f}\n"
            report += f"- PnL Difference: ${perf['pnl_difference']:.2f}\n"
            report += f"- PnL Ratio: {perf['pnl_ratio']:.2f}\n"
            report += f"- Avg PnL Diff/Trade: ${perf['avg_pnl_diff_per_trade']:.2f}\n"
        
    else:  # text format
        report = "TRADE EXECUTION VALIDATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += "SUMMARY\n"
        report += "-" * 20 + "\n"
        report += f"Expected Trades: {summary['expected_count']}\n"
        report += f"Actual Trades: {summary['actual_count']}\n"
        report += f"Matched Trades: {summary.get('matched_count', 0)}\n"
        report += f"Match Rate: {summary['match_rate']:.1%}\n"
        
    return report


def find_missed_opportunities(
    expected_trades: pd.DataFrame,
    actual_trades: pd.DataFrame,
    strategy_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Find expected trades that were not executed.
    
    Args:
        expected_trades: DataFrame of expected trades
        actual_trades: DataFrame of actual trades
        strategy_id: Optional filter by strategy
        
    Returns:
        DataFrame of missed trade opportunities
    """
    validation = validate_execution(expected_trades, actual_trades)
    
    if not validation['unmatched_expected']:
        return pd.DataFrame()
    
    missed = pd.DataFrame(validation['unmatched_expected'])
    
    if strategy_id:
        missed = missed[missed['strategy_id'] == strategy_id]
    
    # Add analysis columns
    if not missed.empty:
        missed['potential_pnl'] = missed['pnl']
        missed['trade_type'] = missed['direction']
        
    return missed.sort_values('entry_bar_idx')