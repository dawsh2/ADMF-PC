#!/usr/bin/env python3
"""
Compare performance of multiple strategies from signal files.
"""

import sys
from pathlib import Path
from src.analytics.signal_reconstruction import SignalReconstructor
import json

def main():
    workspace_dir = "workspaces/tmp/20250611_171158"
    market_data = "data/SPY_1m.csv"
    
    # Find all signal files
    signal_files = list(Path(workspace_dir).glob("signals_strategy_*.json"))
    
    print("=" * 80)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("=" * 80)
    
    all_metrics = []
    
    for signal_file in sorted(signal_files):
        print(f"\nAnalyzing: {signal_file.name}")
        print("-" * 40)
        
        # Reconstruct and analyze
        reconstructor = SignalReconstructor(str(signal_file), market_data)
        report = reconstructor.generate_performance_report()
        
        # Extract key metrics
        metrics = report['performance_metrics']
        metadata = report['metadata']
        
        # Display individual results
        strategy_id = list(metrics['by_strategy'].keys())[0] if metrics['by_strategy'] else 'unknown'
        
        print(f"Strategy ID: {strategy_id}")
        print(f"Total bars: {metadata['total_bars']}")
        print(f"Signal changes: {metadata['total_changes']}")
        print(f"Compression ratio: {metadata.get('compression_ratio', 0):.1%}")
        print(f"\nPerformance:")
        print(f"  Total trades: {metrics['total_trades']}")
        print(f"  Win rate: {metrics['win_rate']*100:.1f}%")
        print(f"  Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"  Avg winner: ${metrics['avg_winner']:.2f}")
        print(f"  Avg loser: ${metrics['avg_loser']:.2f}")
        print(f"  Profit factor: {metrics['profit_factor']:.2f}")
        print(f"  Avg bars held: {metrics['avg_bars_held']:.1f}")
        
        # Store for comparison
        all_metrics.append({
            'strategy': strategy_id,
            'trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'total_pnl': metrics['total_pnl'],
            'profit_factor': metrics['profit_factor'],
            'avg_bars': metrics['avg_bars_held'],
            'signal_freq': metadata['total_changes'] / metadata['total_bars'] if metadata['total_bars'] > 0 else 0
        })
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 80)
    
    # Find best performers
    if all_metrics:
        # Sort by different criteria
        by_win_rate = max(all_metrics, key=lambda x: x['win_rate'])
        by_profit_factor = max(all_metrics, key=lambda x: x['profit_factor'] if x['profit_factor'] != float('inf') else 0)
        by_trades = max(all_metrics, key=lambda x: x['trades'])
        by_pnl = max(all_metrics, key=lambda x: x['total_pnl'])
        
        print(f"\nBest Win Rate: {by_win_rate['strategy']} ({by_win_rate['win_rate']*100:.1f}%)")
        print(f"Best Profit Factor: {by_profit_factor['strategy']} ({by_profit_factor['profit_factor']:.2f})")
        print(f"Most Active: {by_trades['strategy']} ({by_trades['trades']} trades)")
        print(f"Highest P&L: {by_pnl['strategy']} (${by_pnl['total_pnl']:.2f})")
        
        # Strategy characteristics
        print("\nStrategy Characteristics:")
        for m in all_metrics:
            signal_freq_pct = m['signal_freq'] * 100
            print(f"\n{m['strategy']}:")
            print(f"  Trading style: {'Active' if m['trades'] >= 5 else 'Patient'}")
            print(f"  Signal frequency: {signal_freq_pct:.1f}% of bars")
            print(f"  Average hold time: {m['avg_bars']:.1f} bars")
            print(f"  Risk/reward: {m['profit_factor']:.2f} profit factor")

if __name__ == "__main__":
    main()