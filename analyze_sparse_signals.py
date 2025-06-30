#!/usr/bin/env python3
"""
Analyze sparse signal data to verify filters are working.
"""

import pandas as pd
from pathlib import Path
import sys

# Find the latest results directory
results_base = Path("config/keltner/results")
latest_link = results_base / "latest"

if latest_link.exists() and latest_link.is_symlink():
    latest_dir = latest_link.resolve()
else:
    # Get most recent directory
    result_dirs = [d for d in results_base.iterdir() if d.is_dir() and d.name != 'latest']
    if result_dirs:
        latest_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)
    else:
        print("No results directories found")
        exit(1)

print(f"📁 Analyzing results in: {latest_dir}")

# Get all strategy files
traces_dir = latest_dir / "traces" / "keltner_bands"
strategy_files = list(traces_dir.glob("*.parquet"))

print(f"📊 Found {len(strategy_files)} strategy files")

# Analyze signal counts for different strategy groups
strategy_groups = {
    "Baseline (no filter)": list(range(0, 25)),
    "RSI Filter": list(range(25, 61)),
    "Volume Filter": list(range(61, 97)),
    "Combined RSI+Volume": list(range(97, 106)),
    "Directional RSI": list(range(106, 122)),
    "Volatility Filter": list(range(122, 125)),
    "VWAP Filter": list(range(125, 134)),
    "Time Filter": [134],
    "Complex Combinations": list(range(135, 275))
}

print("\n📈 Signal Count Analysis by Filter Type:\n")

for group_name, strategy_indices in strategy_groups.items():
    signal_counts = []
    
    for idx in strategy_indices:
        file_path = traces_dir / f"SPY_5m_compiled_strategy_{idx}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            signal_counts.append(len(df))
    
    if signal_counts:
        avg_signals = sum(signal_counts) / len(signal_counts)
        min_signals = min(signal_counts)
        max_signals = max(signal_counts)
        
        # Calculate reduction vs baseline
        if group_name != "Baseline (no filter)":
            baseline_avg = sum(signal_counts[:25]) / 25 if len(signal_counts) >= 25 else 3000
            reduction = (1 - avg_signals / baseline_avg) * 100
            print(f"{group_name}:")
            print(f"  Average: {avg_signals:.0f} signal changes ({reduction:.1f}% reduction)")
            print(f"  Range: {min_signals} - {max_signals}")
        else:
            baseline_avg = avg_signals
            print(f"{group_name}:")
            print(f"  Average: {avg_signals:.0f} signal changes")
            print(f"  Range: {min_signals} - {max_signals}")

# Detailed comparison of specific examples
print("\n🔍 Detailed Examples:\n")

examples = [
    (0, "Baseline - period=10, mult=1.0"),
    (25, "RSI < 40 - period=10, mult=1.0"),
    (61, "Volume > 1.1x avg - period=10, mult=1.0"),
    (97, "RSI < 50 AND Volume > 1.0x - period=20, mult=2.0"),
    (106, "Directional RSI - period=20, mult=2.0"),
    (134, "Time exclude 12:00-14:30 - period=20, mult=2.0"),
]

for idx, description in examples:
    file_path = traces_dir / f"SPY_5m_compiled_strategy_{idx}.parquet"
    if file_path.exists():
        df = pd.read_parquet(file_path)
        print(f"Strategy {idx}: {description}")
        print(f"  Signal changes: {len(df)}")
        print(f"  Buy entries: {(df['val'] > 0).sum()}")
        print(f"  Sell entries: {(df['val'] < 0).sum()}")
        print(f"  Exit signals: {(df['val'] == 0).sum()}")
        
        if len(df) > 0:
            # Calculate average hold time
            if len(df) > 1:
                avg_bars_between_signals = df['idx'].diff().mean()
                print(f"  Avg bars between signals: {avg_bars_between_signals:.1f}")
        print()

print("✅ Analysis complete!")