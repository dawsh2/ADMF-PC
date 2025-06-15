#!/usr/bin/env python3
"""Analyze which strategies actually produced signals in a workspace."""

import pandas as pd
from pathlib import Path
import sys

def analyze_signals(workspace_path: str):
    """Analyze signal files to see which strategies produced signals."""
    
    workspace = Path(workspace_path)
    signals_dir = workspace / "traces" / "SPY_1m" / "signals"
    
    if not signals_dir.exists():
        print(f"No signals directory found in {workspace_path}")
        return
    
    # Collect all strategy info
    all_strategies = {}
    total_signals = 0
    
    print(f"ANALYZING SIGNALS IN: {workspace_path}")
    print("=" * 80)
    
    # Scan all signal files
    for strategy_type_dir in sorted(signals_dir.iterdir()):
        if strategy_type_dir.is_dir():
            strategy_type = strategy_type_dir.name
            
            for parquet_file in strategy_type_dir.glob("*.parquet"):
                strategy_name = parquet_file.stem
                
                try:
                    df = pd.read_parquet(parquet_file)
                    signal_count = len(df)
                    total_signals += signal_count
                    
                    # Extract parameters from strategy name
                    # e.g., "bollinger_breakout_11_1.5" -> params
                    parts = strategy_name.split('_')
                    
                    all_strategies[strategy_name] = {
                        'type': strategy_type,
                        'file': parquet_file.name,
                        'signals': signal_count,
                        'has_signals': signal_count > 0
                    }
                    
                except Exception as e:
                    print(f"Error reading {parquet_file}: {e}")
    
    # Summary by strategy type
    print("\nSTRATEGY TYPES WITH SIGNALS:")
    strategy_types = {}
    for strat_name, info in all_strategies.items():
        strat_type = info['type']
        if strat_type not in strategy_types:
            strategy_types[strat_type] = {'count': 0, 'with_signals': 0, 'total_signals': 0}
        
        strategy_types[strat_type]['count'] += 1
        if info['has_signals']:
            strategy_types[strat_type]['with_signals'] += 1
            strategy_types[strat_type]['total_signals'] += info['signals']
    
    print(f"{'Strategy Type':<35} {'Total':>8} {'W/Signals':>10} {'Signals':>10}")
    print("-" * 65)
    
    type_count = 0
    for strat_type, stats in sorted(strategy_types.items()):
        if stats['with_signals'] > 0:
            type_count += 1
            print(f"{strat_type:<35} {stats['count']:>8} {stats['with_signals']:>10} {stats['total_signals']:>10}")
    
    # Overall summary
    strategies_with_signals = sum(1 for info in all_strategies.values() if info['has_signals'])
    
    print("\n" + "=" * 65)
    print(f"Total strategy types: {len(strategy_types)}")
    print(f"Strategy types with signals: {type_count}")
    print(f"Total strategy instances: {len(all_strategies)}")
    print(f"Strategies with signals: {strategies_with_signals}")
    print(f"Total signals: {total_signals}")
    
    # Check for missing strategies
    print("\n=== MISSING STRATEGY ANALYSIS ===")
    
    # Expected strategy types from config
    expected_types = [
        'bollinger_breakout', 'donchian_breakout', 'keltner_breakout',
        'stochastic_crossover', 'macd_crossover', 'psar_flip',
        'supertrend', 'chandelier_exit', 'pivot_points',
        'fibonacci_retracement', 'support_resistance_breakout',
        'camarilla_pivot', 'demarker', 'trix_crossover',
        'kama_crossover', 'vidya_crossover', 'fractal_breakout',
        'ichimoku_cloud', 'aroon_crossover', 'vortex_crossover',
        'elder_ray', 'mass_index', 'choppiness_index'
    ]
    
    found_types = set(st.replace('_grid', '') for st in strategy_types.keys())
    missing_types = [t for t in expected_types if t not in found_types]
    
    if missing_types:
        print(f"\nMissing strategy types ({len(missing_types)}):")
        for mt in missing_types:
            print(f"  - {mt}")
    
    # Sample some strategies without signals
    print("\n=== STRATEGIES WITHOUT SIGNALS (sample) ===")
    no_signal_strategies = [(name, info) for name, info in all_strategies.items() if not info['has_signals']]
    
    if no_signal_strategies:
        print(f"Found {len(no_signal_strategies)} strategies with 0 signals")
        print("Sample (first 10):")
        for name, info in no_signal_strategies[:10]:
            print(f"  - {name} (type: {info['type']})")
    
    return all_strategies, total_signals

if __name__ == "__main__":
    # Use a workspace with trace files
    workspace = sys.argv[1] if len(sys.argv) > 1 else "workspaces/expansive_grid_search_fe6e5d3b"
    analyze_signals(workspace)