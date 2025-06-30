#!/usr/bin/env python3
"""
Analyze performance of Keltner strategies with filters applied.
Focus on comparing filtered vs unfiltered strategies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

class FilteredStrategyAnalyzer:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.signals_path = self.workspace_path / "traces" / "keltner_bands"
        self.metadata_path = self.workspace_path / "metadata.json"
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Group strategies by signal count (proxy for filter type)
        self.strategy_groups = self._group_strategies_by_signals()
    
    def _group_strategies_by_signals(self) -> Dict[int, List[str]]:
        """Group strategies by their signal count."""
        groups = {}
        
        for name, comp in self.metadata['components'].items():
            if name.startswith('SPY_5m_compiled_strategy_'):
                signal_count = comp.get('signal_changes', 0)
                if signal_count not in groups:
                    groups[signal_count] = []
                groups[signal_count].append(name)
        
        return groups
    
    def analyze_strategy(self, strategy_file: Path) -> Dict:
        """Analyze a single strategy's performance."""
        try:
            # Load signals
            signals_df = pd.read_parquet(strategy_file)
            signals_df = signals_df.sort_values('idx').reset_index(drop=True)
            
            # Calculate trades and returns
            trades = []
            current_position = None
            
            for i in range(len(signals_df)):
                row = signals_df.iloc[i]
                signal = row['val']
                price = row['px']
                idx = row['idx']
                
                if signal != 0:
                    if current_position is not None:
                        # Close existing position
                        if current_position['direction'] == 'long':
                            ret = np.log(price / current_position['entry_price']) * 10000
                        else:
                            ret = -np.log(price / current_position['entry_price']) * 10000
                        
                        trades.append({
                            'entry_idx': current_position['entry_idx'],
                            'exit_idx': idx,
                            'return_bps': ret,
                            'direction': current_position['direction']
                        })
                    
                    # Open new position
                    current_position = {
                        'entry_idx': idx,
                        'entry_price': price,
                        'direction': 'long' if signal > 0 else 'short'
                    }
                elif signal == 0 and current_position is not None:
                    # Exit signal
                    if current_position['direction'] == 'long':
                        ret = np.log(price / current_position['entry_price']) * 10000
                    else:
                        ret = -np.log(price / current_position['entry_price']) * 10000
                    
                    trades.append({
                        'entry_idx': current_position['entry_idx'],
                        'exit_idx': idx,
                        'return_bps': ret,
                        'direction': current_position['direction']
                    })
                    current_position = None
            
            if not trades:
                return None
            
            # Calculate metrics
            trades_df = pd.DataFrame(trades)
            
            # Apply execution costs
            exec_cost_bps = 0.5
            trades_df['return_bps'] = trades_df['return_bps'] * (1 - exec_cost_bps / 10000)
            
            # Calculate metrics
            total_return = np.exp(trades_df['return_bps'].sum() / 10000) - 1
            
            # Separate long/short
            long_trades = trades_df[trades_df['direction'] == 'long']
            short_trades = trades_df[trades_df['direction'] == 'short']
            
            return {
                'signal_count': len(signals_df),
                'trade_count': len(trades_df),
                'total_return': total_return,
                'avg_return_bps': trades_df['return_bps'].mean(),
                'win_rate': (trades_df['return_bps'] > 0).mean(),
                'sharpe': trades_df['return_bps'].mean() / trades_df['return_bps'].std() if len(trades_df) > 1 else 0,
                'max_win_bps': trades_df['return_bps'].max(),
                'max_loss_bps': trades_df['return_bps'].min(),
                'long_trades': len(long_trades),
                'short_trades': len(short_trades),
                'long_avg_bps': long_trades['return_bps'].mean() if len(long_trades) > 0 else 0,
                'short_avg_bps': short_trades['return_bps'].mean() if len(short_trades) > 0 else 0,
                'long_win_rate': (long_trades['return_bps'] > 0).mean() if len(long_trades) > 0 else 0,
                'short_win_rate': (short_trades['return_bps'] > 0).mean() if len(short_trades) > 0 else 0
            }
        except Exception as e:
            print(f"Error analyzing {strategy_file}: {e}")
            return None
    
    def analyze_all_groups(self):
        """Analyze all strategy groups."""
        print("=== KELTNER FILTERED STRATEGY ANALYSIS ===\n")
        
        # Sort groups by signal count
        sorted_groups = sorted(self.strategy_groups.items(), key=lambda x: x[0])
        
        group_results = []
        
        for signal_count, strategies in sorted_groups:
            print(f"\nAnalyzing group with {signal_count} signals ({len(strategies)} strategies)...")
            
            # Analyze each strategy in the group
            group_metrics = []
            
            for strategy_name in strategies[:5]:  # Sample first 5 from each group
                strategy_num = int(strategy_name.split('_')[-1])
                strategy_file = self.signals_path / f"{strategy_name}.parquet"
                
                if strategy_file.exists():
                    metrics = self.analyze_strategy(strategy_file)
                    if metrics:
                        metrics['strategy_num'] = strategy_num
                        group_metrics.append(metrics)
            
            if group_metrics:
                # Calculate group averages
                df = pd.DataFrame(group_metrics)
                
                group_summary = {
                    'signal_count': signal_count,
                    'strategy_count': len(strategies),
                    'avg_trades': df['trade_count'].mean(),
                    'avg_return_bps': df['avg_return_bps'].mean(),
                    'avg_total_return': df['total_return'].mean(),
                    'avg_win_rate': df['win_rate'].mean(),
                    'avg_sharpe': df['sharpe'].mean(),
                    'avg_long_bps': df['long_avg_bps'].mean(),
                    'avg_short_bps': df['short_avg_bps'].mean(),
                    'long_win_rate': df['long_win_rate'].mean(),
                    'short_win_rate': df['short_win_rate'].mean()
                }
                
                group_results.append(group_summary)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(group_results)
        summary_df = summary_df.sort_values('signal_count')
        
        # Calculate filter effectiveness
        baseline_idx = summary_df['signal_count'].idxmax()
        baseline_return = summary_df.loc[baseline_idx, 'avg_return_bps']
        
        summary_df['filter_reduction'] = (1 - summary_df['signal_count'] / summary_df['signal_count'].max()) * 100
        summary_df['return_improvement'] = ((summary_df['avg_return_bps'] / baseline_return) - 1) * 100
        
        # Print summary table
        print("\n" + "="*120)
        print("FILTER GROUP PERFORMANCE SUMMARY")
        print("="*120)
        print(f"{'Signals':<10} {'Strategies':<12} {'Avg Trades':<12} {'RPT (bps)':<12} {'Win Rate':<10} {'Sharpe':<10} {'Filter %':<10} {'Improve %':<10}")
        print("-"*120)
        
        for _, row in summary_df.iterrows():
            print(f"{row['signal_count']:<10.0f} {row['strategy_count']:<12.0f} "
                  f"{row['avg_trades']:<12.1f} {row['avg_return_bps']:<12.2f} "
                  f"{row['avg_win_rate']*100:<10.1f} {row['avg_sharpe']:<10.2f} "
                  f"{row['filter_reduction']:<10.1f} {row['return_improvement']:<10.1f}")
        
        # Identify best filter group
        best_idx = summary_df['avg_return_bps'].idxmax()
        best_group = summary_df.loc[best_idx]
        
        print(f"\n{'='*60}")
        print("BEST PERFORMING FILTER GROUP")
        print(f"{'='*60}")
        print(f"Signal count: {best_group['signal_count']:.0f} ({best_group['filter_reduction']:.1f}% reduction)")
        print(f"Return per trade: {best_group['avg_return_bps']:.2f} bps")
        print(f"Improvement over baseline: {best_group['return_improvement']:.1f}%")
        print(f"Win rate: {best_group['avg_win_rate']*100:.1f}%")
        print(f"Sharpe ratio: {best_group['avg_sharpe']:.2f}")
        
        # Long/Short analysis
        print(f"\n{'='*60}")
        print("LONG vs SHORT PERFORMANCE BY FILTER GROUP")
        print(f"{'='*60}")
        print(f"{'Signals':<10} {'Long RPT':<12} {'Short RPT':<12} {'Long Win%':<12} {'Short Win%':<12} {'L/S Ratio':<10}")
        print("-"*60)
        
        for _, row in summary_df.iterrows():
            ls_ratio = row['avg_long_bps'] / row['avg_short_bps'] if row['avg_short_bps'] != 0 else np.inf
            print(f"{row['signal_count']:<10.0f} {row['avg_long_bps']:<12.2f} "
                  f"{row['avg_short_bps']:<12.2f} {row['long_win_rate']*100:<12.1f} "
                  f"{row['short_win_rate']*100:<12.1f} {ls_ratio:<10.2f}")
        
        # Save results
        summary_df.to_csv("keltner_filter_group_analysis.csv", index=False)
        print(f"\nDetailed results saved to keltner_filter_group_analysis.csv")
        
        return summary_df
    
    def analyze_specific_strategies(self, strategy_numbers: List[int]):
        """Analyze specific strategies in detail."""
        print(f"\n{'='*60}")
        print("DETAILED ANALYSIS OF SPECIFIC STRATEGIES")
        print(f"{'='*60}")
        
        results = []
        
        for strategy_num in strategy_numbers:
            strategy_name = f"SPY_5m_compiled_strategy_{strategy_num}"
            strategy_file = self.signals_path / f"{strategy_name}.parquet"
            
            if strategy_file.exists():
                metrics = self.analyze_strategy(strategy_file)
                if metrics:
                    # Get signal count from metadata
                    signal_count = self.metadata['components'][strategy_name]['signal_changes']
                    
                    print(f"\nStrategy {strategy_num}:")
                    print(f"  Signals: {signal_count}")
                    print(f"  Trades: {metrics['trade_count']}")
                    print(f"  Return per trade: {metrics['avg_return_bps']:.2f} bps")
                    print(f"  Total return: {metrics['total_return']*100:.1f}%")
                    print(f"  Win rate: {metrics['win_rate']*100:.1f}%")
                    print(f"  Sharpe: {metrics['sharpe']:.2f}")
                    print(f"  Long: {metrics['long_avg_bps']:.2f} bps ({metrics['long_trades']} trades)")
                    print(f"  Short: {metrics['short_avg_bps']:.2f} bps ({metrics['short_trades']} trades)")
                    
                    results.append({
                        'strategy': strategy_num,
                        'signals': signal_count,
                        **metrics
                    })
        
        return pd.DataFrame(results)


def main():
    # Analyze the latest workspace
    workspace = "/Users/daws/ADMF-PC/config/keltner/results/latest"
    
    analyzer = FilteredStrategyAnalyzer(workspace)
    
    # Analyze all filter groups
    summary_df = analyzer.analyze_all_groups()
    
    # Analyze specific interesting strategies
    # Pick one from each signal count group
    interesting_strategies = [
        0,    # Baseline (3262 signals)
        20,   # Heavy filter (47 signals)
        15,   # Strong filter (161 signals)
        21,   # Moderate filter (303 signals)
        4,    # Light filter (2305 signals)
    ]
    
    detailed_df = analyzer.analyze_specific_strategies(interesting_strategies)
    
    # Final recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Find optimal filter level
    optimal_idx = summary_df.loc[summary_df['avg_return_bps'] > 0, 'avg_return_bps'].idxmax()
    optimal = summary_df.loc[optimal_idx]
    
    print(f"\n1. Optimal Filter Level:")
    print(f"   - Signal count: {optimal['signal_count']:.0f} ({optimal['filter_reduction']:.1f}% reduction)")
    print(f"   - Expected performance: {optimal['avg_return_bps']:.2f} bps/trade")
    print(f"   - Trade frequency: {optimal['avg_trades']:.0f} trades")
    
    print(f"\n2. Direction Bias:")
    if optimal['avg_long_bps'] > optimal['avg_short_bps'] * 1.5:
        print(f"   - Strong long bias detected")
        print(f"   - Consider long-only implementation")
    else:
        print(f"   - Balanced long/short performance")
    
    print(f"\n3. Implementation:")
    if optimal['signal_count'] < 100:
        print(f"   - Very restrictive filter (likely master regime filter)")
        print(f"   - Ensure execution quality given low trade frequency")
    elif optimal['signal_count'] < 500:
        print(f"   - Strong filtering (volatility/VWAP based)")
        print(f"   - Good balance of edge and frequency")
    else:
        print(f"   - Moderate filtering")
        print(f"   - Higher frequency but lower edge per trade")


if __name__ == "__main__":
    main()