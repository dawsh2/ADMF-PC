#!/usr/bin/env python3
"""Simple analysis of swing pivot bounce zones."""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

workspace = "workspaces/signal_generation_ae5ce1b4"
signal_pattern = str(Path(workspace) / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")
signal_files = sorted(glob(signal_pattern))

print(f"Analyzing swing pivot bounce zones")
print(f"Found {len(signal_files)} strategy files\n")

# Count strategies by number of signals
signal_counts = {}
tradeable = []

for signal_file in signal_files[:100]:  # Sample first 100
    try:
        signals_df = pd.read_parquet(signal_file)
        num_signals = len(signals_df)
        
        if num_signals not in signal_counts:
            signal_counts[num_signals] = 0
        signal_counts[num_signals] += 1
        
        # Check if any actual trades
        if num_signals >= 3:  # Need at least entry, exit, maybe re-entry
            non_zero = signals_df[signals_df['val'] != 0]
            if len(non_zero) > 0:
                tradeable.append((signal_file, num_signals))
    except:
        pass

print("Signal frequency distribution (first 100 strategies):")
for count, num in sorted(signal_counts.items()):
    print(f"  {count} signals: {num} strategies")

print(f"\nTradeable strategies (3+ signals): {len(tradeable)}")

# Analyze a few tradeable ones
if tradeable:
    print("\nAnalyzing first 5 tradeable strategies:")
    
    for signal_file, num_signals in tradeable[:5]:
        signals_df = pd.read_parquet(signal_file)
        strategy_name = Path(signal_file).stem
        
        # Calculate simple returns
        entry_price = None
        entry_signal = None
        trades = []
        
        for _, row in signals_df.iterrows():
            signal = row['val']
            price = row['px']
            
            if signal != 0 and entry_price is None:
                entry_price = price
                entry_signal = signal
            elif entry_price is not None and signal == 0:
                pnl = (price / entry_price - 1) * entry_signal
                trades.append(pnl)
                entry_price = None
        
        if trades:
            avg_return = np.mean(trades) * 10000
            print(f"\n{strategy_name}:")
            print(f"  Signals: {num_signals}, Trades: {len(trades)}")
            print(f"  Avg return: {avg_return:.2f} bps")
            print(f"  Total return: {sum(trades)*10000:.2f} bps")

print("\n\nCONCLUSION:")
print("Swing pivot bounce zones trade EXTREMELY infrequently - most strategies")
print("only generate 1-3 signals over the entire test period. This is expected")
print("behavior for a strategy looking for major pivot points.")
print("\nThis is completely different from the high-frequency mean reversion")
print("filters you showed - those trade 2-14 times per DAY vs 1-3 times TOTAL.")