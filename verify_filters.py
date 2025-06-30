#!/usr/bin/env python3
"""
Quick verification that filters are being applied.
"""

import pandas as pd
from pathlib import Path
import numpy as np

# Find the latest results directory
results_base = Path("config/keltner/results")
if results_base.exists():
    # Get most recent directory
    result_dirs = [d for d in results_base.iterdir() if d.is_dir() and d.name != 'latest']
    if result_dirs:
        latest_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)
        print(f"Using results from: {latest_dir}")
    else:
        print("No results directories found")
        exit(1)
else:
    print(f"Results directory not found: {results_base}")
    exit(1)

traces_dir = latest_dir / "traces" / "keltner_bands"
if not traces_dir.exists():
    print(f"No traces found in {traces_dir}")
    exit(1)

# Load a few strategies to compare
print("\n📊 Comparing Filter Effects:\n")

# Strategy 0: Baseline (no filter)
baseline_file = traces_dir / "SPY_5m_compiled_strategy_0.parquet"
if baseline_file.exists():
    baseline_df = pd.read_parquet(baseline_file)
    # In sparse format, each row is a signal change
    baseline_changes = len(baseline_df)
    buy_signals = (baseline_df['val'] > 0).sum()
    sell_signals = (baseline_df['val'] < 0).sum()
    flat_signals = (baseline_df['val'] == 0).sum()
    
    print(f"✅ Baseline Strategy (no filter):")
    print(f"   Total signal changes: {baseline_changes}")
    print(f"   Buy signals: {buy_signals}")
    print(f"   Sell signals: {sell_signals}")
    print(f"   Flat signals: {flat_signals}")

# Check a few filtered strategies
filter_examples = [
    (25, "RSI Filter"),
    (61, "Volume Filter"),
    (97, "Combined RSI+Volume"),
    (106, "Directional RSI"),
    (134, "Time Filter"),
]

print("\n📈 Filtered Strategies:")
for idx, filter_name in filter_examples:
    file_path = traces_dir / f"SPY_5m_compiled_strategy_{idx}.parquet"
    if file_path.exists():
        df = pd.read_parquet(file_path)
        total_changes = len(df)
        reduction = (1 - total_changes / baseline_changes) * 100 if baseline_changes > 0 else 0
        print(f"\n{filter_name} (strategy {idx}):")
        print(f"   Total signal changes: {total_changes} ({reduction:.1f}% reduction)")
        print(f"   Buy signals: {(df['val'] > 0).sum()}")
        print(f"   Sell signals: {(df['val'] < 0).sum()}")
        print(f"   Flat signals: {(df['val'] == 0).sum()}")

# Sample some signals to see the actual values
print("\n🔍 Sample Signal Differences:")
if baseline_file.exists():
    baseline_df = pd.read_parquet(baseline_file)
    
    # Compare with an RSI filtered strategy
    rsi_file = traces_dir / "SPY_5m_compiled_strategy_25.parquet"
    if rsi_file.exists():
        rsi_df = pd.read_parquet(rsi_file)
        
        # Compare signal change counts
        print(f"\nBaseline vs RSI Filter:")
        print(f"   Baseline changes: {len(baseline_df)}")
        print(f"   RSI filter changes: {len(rsi_df)}")
        print(f"   Reduction: {(1 - len(rsi_df)/len(baseline_df))*100:.1f}%")
        
        # Show first few signals from each
        print(f"\n   First 5 baseline signals:")
        for _, row in baseline_df.head().iterrows():
            print(f"     Bar {row['idx']}: signal={row['val']}, price={row['px']:.2f}")
        
        print(f"\n   First 5 RSI filtered signals:")
        for _, row in rsi_df.head().iterrows():
            print(f"     Bar {row['idx']}: signal={row['val']}, price={row['px']:.2f}")

print("\n✅ Filter verification complete!")