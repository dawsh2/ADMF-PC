#!/usr/bin/env python3
"""Analyze ALL swing pivot bounce zones strategies."""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

workspace = "workspaces/signal_generation_ae5ce1b4"
signal_pattern = str(Path(workspace) / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")
signal_files = sorted(glob(signal_pattern))

print(f"Analyzing ALL swing pivot bounce zones strategies")
print(f"Found {len(signal_files)} strategy files\n")

# Analyze ALL strategies
signal_counts = {}
all_results = []

for i, signal_file in enumerate(signal_files):
    if i % 100 == 0:
        print(f"Processing strategy {i}/{len(signal_files)}...", end='\r')
    
    try:
        signals_df = pd.read_parquet(signal_file)
        num_signals = len(signals_df)
        
        if num_signals not in signal_counts:
            signal_counts[num_signals] = 0
        signal_counts[num_signals] += 1
        
        # Calculate returns for ALL strategies with signals
        if num_signals >= 3:  # Need at least entry and exit
            strategy_name = Path(signal_file).stem
            strategy_id = int(strategy_name.split('_')[-1])
            
            # Calculate returns
            entry_price = None
            entry_signal = None
            trades = []
            
            for _, row in signals_df.iterrows():
                signal = row['val']
                price = row['px']
                
                if signal != 0 and entry_price is None:
                    entry_price = price
                    entry_signal = signal
                elif entry_price is not None and (signal == 0 or signal == -entry_signal):
                    # Exit or reversal
                    log_return = np.log(price / entry_price) * entry_signal
                    trades.append(log_return)
                    
                    if signal != 0:  # Reversal
                        entry_price = price
                        entry_signal = signal
                    else:  # Exit
                        entry_price = None
            
            if trades:
                trade_returns_bps = [r * 10000 for r in trades]
                edge_bps = np.mean(trade_returns_bps) - 2  # 2bp costs
                
                all_results.append({
                    'strategy_id': int(strategy_id),
                    'num_signals': int(num_signals),
                    'num_trades': int(len(trades)),
                    'edge_bps': float(edge_bps),
                    'total_return_bps': float(sum(trade_returns_bps) - 2 * len(trades))
                })
    except Exception as e:
        print(f"\nError on {signal_file}: {e}")
        continue

print(f"\n\nSignal frequency distribution (ALL {len(signal_files)} strategies):")
for count, num in sorted(signal_counts.items()):
    print(f"  {count} signals: {num} strategies ({num/len(signal_files)*100:.1f}%)")

if all_results:
    df = pd.DataFrame(all_results)
    df = df.sort_values('edge_bps', ascending=False)
    
    print(f"\nStrategies with trades: {len(df)} out of {len(signal_files)}")
    print(f"Average trades per tradeable strategy: {df['num_trades'].mean():.1f}")
    
    print("\n=== TOP 20 STRATEGIES BY EDGE ===")
    print("ID    | Signals | Trades | Edge(bps) | Total Return")
    print("------|---------|--------|-----------|-------------")
    
    for _, row in df.head(20).iterrows():
        print(f"{int(row['strategy_id']):5d} | {int(row['num_signals']):7d} | {int(row['num_trades']):6d} | "
              f"{row['edge_bps']:9.2f} | {row['total_return_bps']:12.0f}")
    
    print("\n=== BOTTOM 10 STRATEGIES ===")
    for _, row in df.tail(10).iterrows():
        print(f"{int(row['strategy_id']):5d} | {int(row['num_signals']):7d} | {int(row['num_trades']):6d} | "
              f"{row['edge_bps']:9.2f} | {row['total_return_bps']:12.0f}")
    
    # Edge distribution
    print(f"\n\nEDGE DISTRIBUTION:")
    print(f"Positive edge: {len(df[df['edge_bps'] > 0])} strategies ({len(df[df['edge_bps'] > 0])/len(df)*100:.1f}%)")
    print(f"Negative edge: {len(df[df['edge_bps'] < 0])} strategies ({len(df[df['edge_bps'] < 0])/len(df)*100:.1f}%)")
    print(f"Best edge: {df['edge_bps'].max():.2f} bps")
    print(f"Worst edge: {df['edge_bps'].min():.2f} bps")
    print(f"Average edge: {df['edge_bps'].mean():.2f} bps")
    
    # Trade frequency
    print(f"\nTRADE FREQUENCY:")
    freq_bins = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 1000)]
    for low, high in freq_bins:
        count = len(df[(df['num_trades'] >= low) & (df['num_trades'] <= high)])
        if count > 0:
            print(f"  {low:3d}-{high:3d} trades: {count:4d} strategies")
    
    # Find high frequency strategies
    high_freq = df[df['num_trades'] >= 10]
    if len(high_freq) > 0:
        print(f"\nHIGH FREQUENCY STRATEGIES (10+ trades):")
        print("ID    | Signals | Trades | Edge(bps)")
        print("------|---------|--------|----------")
        for _, row in high_freq.head(10).iterrows():
            print(f"{int(row['strategy_id']):5d} | {int(row['num_signals']):7d} | {int(row['num_trades']):6d} | {row['edge_bps']:9.2f}")
else:
    print("\nNo strategies generated any complete trades!")