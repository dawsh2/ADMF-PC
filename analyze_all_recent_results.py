#!/usr/bin/env python3
"""Analyze all recent parameter tests"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

# Map result directories to parameters
results_map = [
    ("20250622_212146", "P=10, M=3.0"),
    ("20250622_213055", "P=20, M=3.0"),
    ("20250622_214104", "P=15, M=3.0"),
    ("20250622_214252", "P=23, M=3.0"),
]

# Check if there's a newer one
latest_dir = Path("config/keltner/robust_config/results/latest")
if latest_dir.exists():
    # Get actual directory it points to
    latest_files = list(latest_dir.glob("metadata.json"))
    if latest_files:
        latest_timestamp = latest_dir.resolve().name
        if latest_timestamp not in [r[0] for r in results_map]:
            results_map.append((latest_timestamp, "P=10, M=0.5"))

def analyze_result(timestamp):
    """Quick analysis of a result"""
    result_dir = Path(f"config/keltner/robust_config/results/{timestamp}")
    metadata_file = result_dir / "metadata.json"
    
    if not metadata_file.exists():
        return None
        
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    traces_dir = result_dir / "traces" / "mean_reversion"
    trace_files = list(traces_dir.glob("*.parquet"))
    
    if not trace_files:
        return None
        
    trace_file = trace_files[0]
    df = pd.read_parquet(trace_file)
    
    # Quick performance calc
    trades = 0
    trade_returns = []
    in_trade = False
    
    for i in range(len(df)):
        signal = df.iloc[i]['val']
        price = df.iloc[i]['px']
        
        if not in_trade and signal != 0:
            trades += 1
            in_trade = True
            entry_price = price
            trade_direction = signal
        elif in_trade and signal == 0:
            in_trade = False
            if 'entry_price' in locals():
                if trade_direction > 0:
                    ret = (price - entry_price) / entry_price
                else:
                    ret = (entry_price - price) / entry_price
                trade_returns.append(ret)
    
    total_bars = metadata['total_bars']
    trading_days = total_bars / 78
    
    if trade_returns:
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
        avg_return = np.mean(trade_returns) * 100
        total_return = sum(trade_returns) * 100
        annual_return = (total_return / trading_days) * 252
    else:
        win_rate = avg_return = total_return = annual_return = 0
    
    comp_name = trace_file.stem
    signal_freq = metadata['components'][comp_name]['signal_frequency'] * 100
    
    return {
        'timestamp': timestamp,
        'signal_freq': signal_freq,
        'trades': trades,
        'trades_per_day': trades / trading_days,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'annual_return': annual_return
    }

print("ALL PARAMETER TESTS ON TEST SET")
print("="*80)
print(f"{'Parameters':<15} {'Trades':>8} {'T/Day':>6} {'Win%':>6} {'Avg Ret%':>9} {'Annual%':>8} {'Signal%':>8}")
print("-"*80)

all_results = []
for timestamp, params in results_map:
    result = analyze_result(timestamp)
    if result:
        print(f"{params:<15} {result['trades']:>8} {result['trades_per_day']:>6.2f} "
              f"{result['win_rate']:>6.1f} {result['avg_return']:>9.4f} "
              f"{result['annual_return']:>8.2f} {result['signal_freq']:>8.2f}")
        all_results.append((params, result))

# Find newest result not in our map
newest_dirs = sorted(Path("config/keltner/robust_config/results").glob("2025*"))
if newest_dirs:
    newest = newest_dirs[-1].name
    if newest not in [r[0] for r in results_map]:
        result = analyze_result(newest)
        if result:
            print(f"{'P=10, M=0.5':<15} {result['trades']:>8} {result['trades_per_day']:>6.2f} "
                  f"{result['win_rate']:>6.1f} {result['avg_return']:>9.4f} "
                  f"{result['annual_return']:>8.2f} {result['signal_freq']:>8.2f}")

print("\n" + "="*80)
print("SUMMARY:")
if all_results:
    best = max(all_results, key=lambda x: x[1]['annual_return'])
    print(f"Best performer: {best[0]} with {best[1]['annual_return']:.2f}% annual return")
    
    # Check if testing tighter bands
    print("\nNOTE: Lower multipliers = tighter bands (more sensitive)")
    print("Original analysis tested M=[1.0, 1.5, 2.0, 2.5, 3.0]")
    print("M=0.5 is MUCH tighter than anything originally tested!")