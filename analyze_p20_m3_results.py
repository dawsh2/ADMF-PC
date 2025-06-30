#!/usr/bin/env python3
"""Analyze the truly robust parameters (Period=20, Mult=3.0) on test set"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

# Find results
old_result = Path("config/keltner/robust_config/results/20250622_212146")  # P=10, M=3
new_result = Path("config/keltner/robust_config/results/20250622_213055")  # P=20, M=3

print("TRULY ROBUST PARAMETERS (P=20, M=3.0) - TEST SET PERFORMANCE")
print("="*80)

def analyze_result(result_dir, param_desc):
    """Analyze a single result directory"""
    metadata_file = result_dir / "metadata.json"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Find trace file
    traces_dir = result_dir / "traces" / "mean_reversion"
    trace_files = list(traces_dir.glob("*.parquet"))
    
    if not trace_files:
        return None
        
    trace_file = trace_files[0]
    df = pd.read_parquet(trace_file)
    
    # Analyze performance
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
                if trade_direction > 0:  # Long
                    ret = (price - entry_price) / entry_price
                else:  # Short
                    ret = (entry_price - price) / entry_price
                trade_returns.append(ret)
    
    # Calculate metrics
    total_bars = metadata['total_bars']
    bars_per_day = 78  # 6.5 hours * 12 (5-min bars)
    trading_days = total_bars / bars_per_day
    
    if trade_returns:
        avg_return = np.mean(trade_returns) * 100
        total_return = sum(trade_returns) * 100
        win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
        
        # Compound return
        cumulative = 1.0
        for r in trade_returns:
            cumulative *= (1 + r)
        compound_return = (cumulative - 1) * 100
        
        # Annualized
        annual_simple = (total_return / trading_days) * 252
        years = trading_days / 252
        annual_compound = ((cumulative ** (1/years)) - 1) * 100 if years > 0 else 0
    else:
        avg_return = total_return = win_rate = compound_return = annual_simple = annual_compound = 0
    
    # Get component info
    comp_name = trace_file.stem
    signal_freq = metadata['components'][comp_name]['signal_frequency'] * 100
    signal_changes = metadata['components'][comp_name]['signal_changes']
    
    return {
        'params': param_desc,
        'total_bars': total_bars,
        'signal_changes': signal_changes,
        'signal_freq': signal_freq,
        'trades': trades,
        'trades_per_day': trades / trading_days if trading_days > 0 else 0,
        'avg_return': avg_return,
        'total_return': total_return,
        'compound_return': compound_return,
        'win_rate': win_rate,
        'annual_simple': annual_simple,
        'annual_compound': annual_compound
    }

# Analyze both parameter sets
results = []

old_data = analyze_result(old_result, "P=10, M=3.0 (bad)")
if old_data:
    results.append(old_data)

new_data = analyze_result(new_result, "P=20, M=3.0 (robust)")
if new_data:
    results.append(new_data)

# Display comparison
if len(results) == 2:
    print(f"{'Metric':<25} {'P=10,M=3 (bad)':>18} {'P=20,M=3 (robust)':>18} {'Improvement':>15}")
    print("-"*80)
    
    metrics = [
        ('Signal changes', 'signal_changes', 'd'),
        ('Signal frequency %', 'signal_freq', '.2f'),
        ('Total trades', 'trades', 'd'),
        ('Trades per day', 'trades_per_day', '.2f'),
        ('Win rate %', 'win_rate', '.1f'),
        ('Avg return/trade %', 'avg_return', '.4f'),
        ('Total return %', 'total_return', '.2f'),
        ('Compound return %', 'compound_return', '.2f'),
        ('Annual return (simple) %', 'annual_simple', '.2f'),
        ('Annual return (compound) %', 'annual_compound', '.2f')
    ]
    
    for label, key, fmt in metrics:
        old_val = results[0][key]
        new_val = results[1][key]
        
        if key in ['win_rate', 'avg_return', 'total_return', 'compound_return', 'annual_simple', 'annual_compound']:
            diff = new_val - old_val
            diff_str = f"{diff:+.2f}"
        else:
            diff_str = "N/A"
            
        print(f"{label:<25} {old_val:>18{fmt}} {new_val:>18{fmt}} {diff_str:>15}")
    
    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)
    
    if results[1]['annual_compound'] > 0:
        print("✅ TRULY ROBUST parameters ARE PROFITABLE on test set!")
        print(f"   Annual return: {results[1]['annual_compound']:.2f}%")
        print(f"   Win rate: {results[1]['win_rate']:.1f}%")
        print(f"   Much better than P=10,M=3 which lost {results[0]['annual_compound']:.2f}%")
    else:
        print("❌ Even robust parameters struggle on test set")
        
    # Compare to training expectations
    print("\nTRAINING vs TEST COMPARISON:")
    print("Training analysis showed P=20,M=3:")
    print("  - 100% profitable across all filters")
    print("  - 0.007% average return per trade")
    print("  - Worst case: 0.000% (breakeven)")
    print(f"\nTest results: {results[1]['avg_return']:.4f}% per trade")
    
else:
    print("ERROR: Could not analyze both results")
    
# Check if this is train or test data
if results and results[-1]['total_bars'] == 20768:
    print("\n✓ Confirmed: This is TEST SET data (20,768 bars)")
elif results and results[-1]['total_bars'] == 16614:
    print("\n✓ Confirmed: This is TRAIN SET data (16,614 bars)")