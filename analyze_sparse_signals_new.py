#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

def analyze_sparse_signals(workspace_path):
    """Analyze sparse signal traces to count signal changes"""
    
    workspace_path = Path(workspace_path)
    signals_dir = workspace_path / "traces/SPY_1m_1m/signals/swing_pivot_bounce_zones"
    
    if not signals_dir.exists():
        print(f"No signals directory found at {signals_dir}")
        return
    
    # Get all parquet files
    parquet_files = list(signals_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} signal files (strategies)")
    
    # Collect all signal changes across all strategies
    all_signal_changes = []
    strategy_signal_counts = defaultdict(int)
    
    # Analyze each file (each represents a strategy)
    for i, file in enumerate(parquet_files):
        df = pd.read_parquet(file)
        
        # Extract strategy index from filename
        strategy_idx = int(file.stem.split('_')[-1])
        
        # In sparse format, each row is a signal change
        num_signals = len(df)
        strategy_signal_counts[strategy_idx] = num_signals
        
        if i < 5:  # Show details for first 5
            print(f"\nStrategy {strategy_idx} ({file.name}):")
            print(f"  Number of signal changes: {num_signals}")
            if num_signals > 0:
                print(f"  Signal values: {sorted(df['val'].unique())}")
                print(f"  Time range: {df['ts'].min()} to {df['ts'].max()}")
        
        # Collect all signal changes with their timestamps
        for _, row in df.iterrows():
            all_signal_changes.append({
                'ts': row['ts'],
                'strategy': strategy_idx,
                'signal': row['val'],
                'idx': row['idx']
            })
    
    # Sort all signal changes by timestamp
    all_signal_changes.sort(key=lambda x: x['ts'])
    
    print(f"\n--- Summary ---")
    print(f"Total strategies: {len(parquet_files)}")
    print(f"Total signal changes across all strategies: {len(all_signal_changes)}")
    print(f"Strategies with signals: {sum(1 for count in strategy_signal_counts.values() if count > 0)}")
    print(f"Average signals per strategy: {len(all_signal_changes) / len(parquet_files):.2f}")
    
    # Count strategies by signal count
    signal_distribution = defaultdict(int)
    for count in strategy_signal_counts.values():
        signal_distribution[count] += 1
    
    print(f"\nSignal count distribution:")
    for count in sorted(signal_distribution.keys())[:10]:
        print(f"  {count} signals: {signal_distribution[count]} strategies")
    
    # Expected orders calculation
    print(f"\n--- Order Expectation ---")
    print(f"If each signal change triggers an order:")
    print(f"  Expected orders ≈ {len(all_signal_changes)}")
    print(f"\nNote: Actual orders depend on:")
    print(f"  - Signal aggregation across strategies")
    print(f"  - Position limits and risk management")
    print(f"  - Market hours and trading constraints")
    
    # Show time distribution of signals
    if all_signal_changes:
        df_signals = pd.DataFrame(all_signal_changes)
        df_signals['ts'] = pd.to_datetime(df_signals['ts'])
        df_signals['date'] = df_signals['ts'].dt.date
        
        print(f"\nSignals by date (first 10 days):")
        daily_counts = df_signals.groupby('date').size()
        for date, count in daily_counts.head(10).items():
            print(f"  {date}: {count} signals")

if __name__ == "__main__":
    workspace = sys.argv[1] if len(sys.argv) > 1 else "workspaces/signal_generation_20250622_095959"
    analyze_sparse_signals(workspace)