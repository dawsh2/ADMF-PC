#!/usr/bin/env python3
"""Analyze swing pivot bounce zones workspace a2d31737."""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

workspace = "workspaces/signal_generation_a2d31737"
signal_pattern = str(Path(workspace) / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")
signal_files = sorted(glob(signal_pattern))

print(f"Analyzing workspace: {workspace}")
print(f"Found {len(signal_files)} strategy files\n")

# Analyze all strategies
all_results = []
signal_freq_dist = {}

for i, signal_file in enumerate(signal_files):
    if i % 100 == 0:
        print(f"Processing strategy {i}/{len(signal_files)}...", end='\r')
    
    try:
        signals_df = pd.read_parquet(signal_file)
        num_signals = len(signals_df)
        
        # Track signal frequency
        freq_bucket = (num_signals // 10) * 10  # Round to nearest 10
        if freq_bucket not in signal_freq_dist:
            signal_freq_dist[freq_bucket] = 0
        signal_freq_dist[freq_bucket] += 1
        
        if num_signals < 10:  # Skip very low signal strategies
            continue
            
        # Extract strategy ID
        strategy_name = Path(signal_file).stem
        parts = strategy_name.split('_')
        strategy_id = int(parts[-1])
        
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
            
            # Time analysis
            first_ts = pd.to_datetime(signals_df['ts'].iloc[0])
            last_ts = pd.to_datetime(signals_df['ts'].iloc[-1])
            trading_days = (last_ts - first_ts).days or 1
            trades_per_day = len(trades) / trading_days
            
            all_results.append({
                'strategy_id': strategy_id,
                'num_signals': num_signals,
                'num_trades': len(trades),
                'edge_bps': edge_bps,
                'total_return_bps': sum(trade_returns_bps) - 2 * len(trades),
                'trades_per_day': trades_per_day,
                'annual_trades': trades_per_day * 252
            })
    except Exception as e:
        continue

print(f"\n\n=== WORKSPACE {workspace.split('/')[-1]} ANALYSIS ===\n")

# Signal frequency distribution
print("SIGNAL FREQUENCY DISTRIBUTION:")
for bucket in sorted(signal_freq_dist.keys()):
    print(f"  {bucket:4d}-{bucket+9:4d} signals: {signal_freq_dist[bucket]:4d} strategies")

if all_results:
    df = pd.DataFrame(all_results)
    df = df.sort_values('edge_bps', ascending=False)
    
    print(f"\nStrategies analyzed: {len(df)} (with 10+ signals)")
    print(f"Average trades per strategy: {df['num_trades'].mean():.1f}")
    print(f"Average trades per day: {df['trades_per_day'].mean():.2f}")
    
    # Performance summary
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"Positive edge: {len(df[df['edge_bps'] > 0])} strategies ({len(df[df['edge_bps'] > 0])/len(df)*100:.1f}%)")
    print(f"Best edge: {df['edge_bps'].max():.2f} bps")
    print(f"Worst edge: {df['edge_bps'].min():.2f} bps")
    print(f"Average edge: {df['edge_bps'].mean():.2f} bps")
    
    # Top performers
    print("\n=== TOP 20 STRATEGIES ===")
    print("ID    | Signals | Trades | Edge(bps) | Trades/Day | Annual")
    print("------|---------|--------|-----------|------------|-------")
    
    for _, row in df.head(20).iterrows():
        print(f"{int(row['strategy_id']):5d} | {int(row['num_signals']):7d} | {int(row['num_trades']):6d} | "
              f"{row['edge_bps']:9.2f} | {row['trades_per_day']:10.2f} | {int(row['annual_trades']):6d}")
    
    # High frequency strategies
    high_freq = df[df['trades_per_day'] >= 0.5]  # At least 0.5 trades/day
    if len(high_freq) > 0:
        print(f"\n=== HIGH FREQUENCY STRATEGIES (0.5+ trades/day) ===")
        print(f"Found {len(high_freq)} strategies")
        print("\nTop 10 by edge:")
        print("ID    | Edge(bps) | Trades/Day | Annual | Total Trades")
        print("------|-----------|------------|--------|-------------")
        for _, row in high_freq.sort_values('edge_bps', ascending=False).head(10).iterrows():
            print(f"{int(row['strategy_id']):5d} | {row['edge_bps']:9.2f} | {row['trades_per_day']:10.2f} | "
                  f"{int(row['annual_trades']):6d} | {int(row['num_trades']):12d}")
    
    # Compare to previous workspace
    print("\n\n=== COMPARISON TO PREVIOUS WORKSPACE (ae5ce1b4) ===")
    print("Previous workspace (ae5ce1b4):")
    print("  - 85% of strategies had only 1 signal")
    print("  - Maximum 4 trades per strategy")
    print("  - Average edge: -2.62 bps")
    print("  - Best edge: 4.88 bps")
    print("\nThis workspace (a2d31737):")
    print(f"  - Much higher signal frequency (see distribution above)")
    print(f"  - Average {df['num_trades'].mean():.0f} trades per strategy")
    print(f"  - Average edge: {df['edge_bps'].mean():.2f} bps")
    print(f"  - Best edge: {df['edge_bps'].max():.2f} bps")
    print(f"  - {len(high_freq)} strategies with 0.5+ trades/day")
    
else:
    print("\nNo strategies with sufficient signals found!")