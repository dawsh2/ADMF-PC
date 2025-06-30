#!/usr/bin/env python3
"""Analyze P=15, M=3.0 results"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

# Results directories
p10_result = Path("config/keltner/robust_config/results/20250622_212146")  # P=10, M=3
p20_result = Path("config/keltner/robust_config/results/20250622_213055")  # P=20, M=3  
p15_result = Path("config/keltner/robust_config/results/20250622_214104")  # P=15, M=3

print("PARAMETER COMPARISON: P=10 vs P=20 vs P=15 (all M=3.0)")
print("="*80)

def analyze_result(result_dir, param_desc):
    """Quick analysis of a result"""
    metadata_file = result_dir / "metadata.json"
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
        
        # Separate long/short
        long_count = (df['val'] > 0).sum()
        short_count = (df['val'] < 0).sum()
    else:
        win_rate = avg_return = total_return = annual_return = 0
        long_count = short_count = 0
    
    comp_name = trace_file.stem
    signal_freq = metadata['components'][comp_name]['signal_frequency'] * 100
    
    return {
        'params': param_desc,
        'signal_freq': signal_freq,
        'trades': trades,
        'trades_per_day': trades / trading_days,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'annual_return': annual_return,
        'long_signals': long_count,
        'short_signals': short_count
    }

# Analyze all three
results = []
for result_dir, desc in [(p10_result, "P=10, M=3.0"), 
                         (p20_result, "P=20, M=3.0"),
                         (p15_result, "P=15, M=3.0")]:
    data = analyze_result(result_dir, desc)
    if data:
        results.append(data)

# Display comparison
print(f"{'Parameters':<15} {'Trades':>8} {'T/Day':>6} {'Win%':>6} {'Avg Ret%':>9} {'Annual%':>8} {'Long':>6} {'Short':>6}")
print("-"*80)

for r in results:
    print(f"{r['params']:<15} {r['trades']:>8} {r['trades_per_day']:>6.2f} "
          f"{r['win_rate']:>6.1f} {r['avg_return']:>9.4f} {r['annual_return']:>8.2f} "
          f"{r['long_signals']:>6} {r['short_signals']:>6}")

# Check which dataset
if results and results[-1]['trades'] > 50:
    print("\n✓ This appears to be TEST SET data (higher trade count)")

# Training expectations from our analysis
print("\n" + "="*80)
print("TRAINING EXPECTATIONS (from comprehensive analysis):")
print("="*80)
print("P=10, M=3.0: 100% profitable filters, 0.058% avg return")
print("P=20, M=3.0: 100% profitable filters, 0.007% avg return")  
print("P=15, M=3.0: 100% profitable filters, 0.012% avg return")

if len(results) >= 3:
    print(f"\nP=15, M=3.0 TEST RESULT: {results[2]['annual_return']:.2f}% annual")
    if results[2]['annual_return'] > results[1]['annual_return']:
        print("✅ P=15 outperforms P=20!")
    else:
        print("❌ P=15 underperforms P=20")