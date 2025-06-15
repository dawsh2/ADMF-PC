#!/usr/bin/env python3
"""Count total signals from a grid search run."""

import pandas as pd
from pathlib import Path
import sys

def count_signals(workspace_path: str):
    """Count total signals across all strategies."""
    
    workspace = Path(workspace_path)
    signals_dir = workspace / "traces" / "SPY_1m" / "signals"
    
    if not signals_dir.exists():
        print(f"No signals directory found in {workspace_path}")
        return
    
    total_signals = 0
    total_strategies = 0
    strategies_with_signals = 0
    strategy_summary = {}
    
    print(f"COUNTING SIGNALS IN: {workspace_path}")
    print("=" * 80)
    
    for strategy_type_dir in sorted(signals_dir.iterdir()):
        if strategy_type_dir.is_dir():
            strategy_type = strategy_type_dir.name
            type_signals = 0
            type_strategies = 0
            
            for parquet_file in strategy_type_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(parquet_file)
                    signals = len(df)
                    type_signals += signals
                    type_strategies += 1
                    if signals > 0:
                        strategies_with_signals += 1
                except Exception as e:
                    print(f"Error reading {parquet_file}: {e}")
            
            total_signals += type_signals
            total_strategies += type_strategies
            strategy_summary[strategy_type] = {
                'strategies': type_strategies,
                'signals': type_signals,
                'avg_per_strategy': type_signals / type_strategies if type_strategies > 0 else 0
            }
    
    # Print summary
    print(f"\nSTRATEGY TYPE SUMMARY:")
    print(f"{'Strategy Type':<30} {'Strategies':>12} {'Signals':>12} {'Avg/Strategy':>12}")
    print("-" * 68)
    
    for strategy_type, stats in sorted(strategy_summary.items()):
        if stats['signals'] > 0:
            print(f"{strategy_type:<30} {stats['strategies']:>12} {stats['signals']:>12} {stats['avg_per_strategy']:>12.1f}")
    
    print("-" * 68)
    print(f"{'TOTAL':<30} {total_strategies:>12} {total_signals:>12} {total_signals/total_strategies if total_strategies > 0 else 0:>12.1f}")
    
    print(f"\n=== RESULTS ===")
    print(f"Total strategies: {total_strategies}")
    print(f"Strategies with signals: {strategies_with_signals}")
    print(f"Success rate: {strategies_with_signals}/{total_strategies} = {strategies_with_signals/total_strategies*100:.1f}%")
    print(f"Total signals generated: {total_signals}")
    
    # Check if this matches expected numbers
    if total_strategies == 882:
        print(f"\n✅ Correct total strategy count (882)")
    else:
        print(f"\n⚠️  Expected 882 strategies, found {total_strategies}")
    
    if strategies_with_signals == 330:
        print(f"✅ Matches previous report (330 strategies with signals)")
    else:
        print(f"⚠️  Previous report had 330 strategies with signals, now {strategies_with_signals}")

if __name__ == "__main__":
    workspace = sys.argv[1] if len(sys.argv) > 1 else "workspaces/expansive_grid_search_fe6e5d3b"
    count_signals(workspace)