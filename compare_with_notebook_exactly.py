#!/usr/bin/env python3
"""Compare our results with notebook exactly."""

import pandas as pd
from pathlib import Path

print("=== Comparing with Notebook Results ===")

# Notebook results (from previous-results.ipynb)
notebook_stats = {
    'total_trades': 416,
    'stop_loss_exits': 175,  # These are the ones we want to prevent re-entry
    'signal_exits': 241,     # Normal exits
    'total_return': 10.27,
    'win_rate': 91.1
}

print("Notebook results:")
for k, v in notebook_stats.items():
    print(f"  {k}: {v}")

# Our results
results_dir = Path("config/bollinger/results/latest")
trades_file = results_dir / "traces/events/portfolio/trades.parquet"

if trades_file.exists():
    trades = pd.read_parquet(trades_file)
    
    # Calculate our stats
    our_stats = {
        'total_trades': len(trades),
        'stop_loss_exits': len(trades[trades['exit_type'] == 'stop_loss']),
        'take_profit_exits': len(trades[trades['exit_type'] == 'take_profit']),
        'signal_exits': len(trades[trades['exit_type'] == 'signal']),
        'other_exits': len(trades[~trades['exit_type'].isin(['stop_loss', 'take_profit', 'signal'])])
    }
    
    print("\n\nOur results:")
    for k, v in our_stats.items():
        print(f"  {k}: {v}")
    
    # The math
    print("\n\nAnalysis:")
    print(f"Notebook: 175 stop losses that should NOT re-enter immediately")
    print(f"Expected trades if exit memory works: 416 - 175 = 241")
    print(f"Our trades: {our_stats['total_trades']}")
    print(f"Difference: {our_stats['total_trades'] - 241}")
    
    # More detailed breakdown
    if our_stats['total_trades'] > 416:
        extra_trades = our_stats['total_trades'] - 416
        print(f"\nWe have {extra_trades} MORE trades than the notebook!")
        print("This suggests:")
        print("1. Exit memory is still not working properly")
        print("2. OR we're creating duplicate exit signals")
        print("3. OR the OHLC logic is causing more exits than expected")
    
    # Check returns to see if performance improved
    trades['return_pct'] = ((trades['exit_price'] - trades['entry_price']) / trades['entry_price']) * 100
    total_return = trades['return_pct'].sum()
    win_rate = (trades['return_pct'] > 0).mean() * 100
    
    print(f"\n\nPerformance comparison:")
    print(f"Notebook return: {notebook_stats['total_return']:.2f}%")
    print(f"Our return: {total_return:.2f}%")
    print(f"Notebook win rate: {notebook_stats['win_rate']:.1f}%")
    print(f"Our win rate: {win_rate:.1f}%")

print("\n\n=== Key Insight ===")
print("The notebook shows 416 trades with 175 stop losses.")
print("If exit memory worked perfectly, we'd see 241 trades.")
print("But we're seeing 463 trades - even MORE than before!")
print("\nThis suggests the OHLC fix might be causing MORE problems.")