#!/usr/bin/env python3
"""
Analyze filter effects on signal generation.

This script compares signals from different strategy configurations
to verify that filters are actually affecting the output.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def load_strategy_signals(parquet_file):
    """Load signals from a parquet file."""
    df = pd.read_parquet(parquet_file)
    return df

def analyze_filter_effects(results_dir):
    """Analyze how filters affect signal generation."""
    results_path = Path(results_dir)
    traces_dir = results_path / "traces" / "keltner_bands"
    
    if not traces_dir.exists():
        print(f"❌ No traces found in {traces_dir}")
        return
    
    # Load metadata to understand strategy configurations
    metadata_file = results_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Group strategies by their base parameters (period, multiplier)
    strategy_groups = defaultdict(list)
    
    # Load all strategy files
    all_strategies = {}
    for parquet_file in traces_dir.glob("*.parquet"):
        strategy_name = parquet_file.stem
        signals_df = load_strategy_signals(parquet_file)
        all_strategies[strategy_name] = signals_df
        
        # Extract strategy number
        if "compiled_strategy_" in strategy_name:
            strategy_num = int(strategy_name.split("_")[-1])
            # Group by base parameters (assuming first 25 are baseline)
            base_group = strategy_num % 25
            strategy_groups[base_group].append((strategy_num, strategy_name, signals_df))
    
    print(f"📊 Loaded {len(all_strategies)} strategy configurations")
    
    # Analyze signal differences
    print("\n🔍 Analyzing Filter Effects\n")
    
    # Compare baseline (no filter) with filtered versions
    baseline_strategies = [(name, df) for name, df in all_strategies.items() 
                          if int(name.split("_")[-1]) < 25]
    
    if baseline_strategies:
        print("📈 Baseline Strategies (No Filters):")
        for name, df in baseline_strategies[:5]:  # Show first 5
            total_signals = len(df)
            buy_signals = (df['signal'] > 0).sum()
            sell_signals = (df['signal'] < 0).sum()
            print(f"  {name}: {total_signals} bars, {buy_signals} buys, {sell_signals} sells")
    
    # Analyze specific filter types
    analyze_filter_type(all_strategies, "RSI Filter", range(25, 61))  # RSI filter strategies
    analyze_filter_type(all_strategies, "Volume Filter", range(61, 97))  # Volume filter strategies
    analyze_filter_type(all_strategies, "Combined RSI+Volume", range(97, 106))  # Combined filters
    analyze_filter_type(all_strategies, "Directional RSI", range(106, 122))  # Directional filters
    
    # Compare signal reduction
    print("\n📉 Signal Reduction Analysis:")
    compare_signal_reduction(all_strategies)
    
    # Detailed comparison of a few strategies
    print("\n🔬 Detailed Comparison (First 5 strategies):")
    for i in range(min(5, len(all_strategies) // 10)):
        compare_strategy_pair(all_strategies, i, i + 25)

def analyze_filter_type(strategies, filter_name, strategy_range):
    """Analyze strategies with a specific filter type."""
    print(f"\n📊 {filter_name} Strategies:")
    
    filtered_strategies = []
    for i in strategy_range:
        strategy_name = f"SPY_5m_compiled_strategy_{i}"
        if strategy_name in strategies:
            filtered_strategies.append((strategy_name, strategies[strategy_name]))
    
    if not filtered_strategies:
        print(f"  No strategies found in range {strategy_range}")
        return
    
    # Calculate average signal reduction
    total_signals = []
    buy_signals = []
    sell_signals = []
    
    for name, df in filtered_strategies[:10]:  # Analyze first 10
        total_signals.append(len(df[df['signal'] != 0]))
        buy_signals.append((df['signal'] > 0).sum())
        sell_signals.append((df['signal'] < 0).sum())
    
    if total_signals:
        avg_total = np.mean(total_signals)
        avg_buys = np.mean(buy_signals)
        avg_sells = np.mean(sell_signals)
        print(f"  Average: {avg_total:.1f} signals ({avg_buys:.1f} buys, {avg_sells:.1f} sells)")
        print(f"  Range: {min(total_signals)}-{max(total_signals)} total signals")

def compare_signal_reduction(strategies):
    """Compare signal counts across different filter configurations."""
    baseline_signals = []
    filtered_signals = defaultdict(list)
    
    # Baseline strategies (0-24)
    for i in range(25):
        name = f"SPY_5m_compiled_strategy_{i}"
        if name in strategies:
            df = strategies[name]
            baseline_signals.append((df['signal'] != 0).sum())
    
    # Filter categories
    filter_ranges = {
        "RSI": range(25, 61),
        "Volume": range(61, 97),
        "Combined": range(97, 106),
        "Directional": range(106, 122),
        "Volatility": range(122, 125),
        "VWAP": range(125, 134),
        "Time": range(134, 135),
    }
    
    for filter_type, strategy_range in filter_ranges.items():
        for i in strategy_range:
            name = f"SPY_5m_compiled_strategy_{i}"
            if name in strategies:
                df = strategies[name]
                filtered_signals[filter_type].append((df['signal'] != 0).sum())
    
    # Print comparison
    if baseline_signals:
        baseline_avg = np.mean(baseline_signals)
        print(f"\nBaseline (no filter): {baseline_avg:.1f} signals on average")
        
        for filter_type, signals in filtered_signals.items():
            if signals:
                avg_signals = np.mean(signals)
                reduction = (1 - avg_signals / baseline_avg) * 100
                print(f"{filter_type:12s}: {avg_signals:.1f} signals ({reduction:.1f}% reduction)")

def compare_strategy_pair(strategies, idx1, idx2):
    """Compare two specific strategies to show filter effect."""
    name1 = f"SPY_5m_compiled_strategy_{idx1}"
    name2 = f"SPY_5m_compiled_strategy_{idx2}"
    
    if name1 not in strategies or name2 not in strategies:
        return
    
    df1 = strategies[name1]
    df2 = strategies[name2]
    
    # Ensure same index for comparison
    common_index = df1.index.intersection(df2.index)
    df1 = df1.loc[common_index]
    df2 = df2.loc[common_index]
    
    # Count differences
    signals1 = (df1['signal'] != 0).sum()
    signals2 = (df2['signal'] != 0).sum()
    
    # Find where signals differ
    signal_diff = (df1['signal'] != df2['signal']).sum()
    
    print(f"\n{name1} vs {name2}:")
    print(f"  Strategy 1: {signals1} signals")
    print(f"  Strategy 2: {signals2} signals")
    print(f"  Difference: {abs(signals1 - signals2)} signals")
    print(f"  Bars with different signals: {signal_diff}")

def main():
    """Main analysis function."""
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Default to latest results
        results_dir = "config/keltner/results/latest"
    
    print(f"🔍 Analyzing filter effects in: {results_dir}\n")
    analyze_filter_effects(results_dir)

if __name__ == "__main__":
    main()