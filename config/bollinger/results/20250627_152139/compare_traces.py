#!/usr/bin/env python3
"""Compare signal traces between notebook and latest results"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the signal trace from latest results
latest_signals_path = Path("../latest/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
latest_signals = pd.read_parquet(latest_signals_path)

print("=== LATEST SYSTEM RESULTS ===")
print(f"Total signals: {len(latest_signals)}")
print(f"Signal values: {latest_signals['val'].value_counts().to_dict()}")
print(f"Date range: {latest_signals['ts'].min()} to {latest_signals['ts'].max()}")
print(f"Price range: ${latest_signals['px'].min():.2f} to ${latest_signals['px'].max():.2f}")

# Show first few signals
print("\nFirst 10 signals:")
print(latest_signals[['idx', 'ts', 'val', 'px']].head(10))

# Show last few signals  
print("\nLast 10 signals:")
print(latest_signals[['idx', 'ts', 'val', 'px']].tail(10))

# Check signal distribution
print("\n=== SIGNAL DISTRIBUTION ===")
print("By signal value:")
print(latest_signals['val'].value_counts())

# Load the traces from current run (notebook)
current_signals_path = Path("traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
if current_signals_path.exists():
    current_signals = pd.read_parquet(current_signals_path)
    
    print("\n=== CURRENT RUN (NOTEBOOK) ===")
    print(f"Total signals: {len(current_signals)}")
    print(f"Signal values: {current_signals['val'].value_counts().to_dict()}")
    print(f"Date range: {current_signals['ts'].min()} to {current_signals['ts'].max()}")
    
    # Compare signals
    print("\n=== SIGNAL COMPARISON ===")
    
    # Check if they have the same indices
    common_idx = set(latest_signals['idx']).intersection(set(current_signals['idx']))
    print(f"Common indices: {len(common_idx)}")
    
    if len(common_idx) > 0:
        # Compare signals at common indices
        latest_subset = latest_signals[latest_signals['idx'].isin(common_idx)].set_index('idx')
        current_subset = current_signals[current_signals['idx'].isin(common_idx)].set_index('idx')
        
        # Find differences
        signal_diffs = latest_subset['val'] != current_subset['val']
        if signal_diffs.any():
            print(f"\nFound {signal_diffs.sum()} signal differences!")
            diff_idx = signal_diffs[signal_diffs].index[:10]  # Show first 10
            
            for idx in diff_idx:
                print(f"  Index {idx}: latest={latest_subset.loc[idx, 'val']}, current={current_subset.loc[idx, 'val']}")
        else:
            print("\nNo signal differences found at common indices")
            
        # Check price differences
        price_diffs = abs(latest_subset['px'] - current_subset['px']) > 0.01
        if price_diffs.any():
            print(f"\nFound {price_diffs.sum()} price differences > $0.01!")
            diff_idx = price_diffs[price_diffs].index[:5]
            
            for idx in diff_idx:
                print(f"  Index {idx}: latest=${latest_subset.loc[idx, 'px']:.2f}, current=${current_subset.loc[idx, 'px']:.2f}")
                
else:
    print("\nCurrent run signal file not found!")

# Check metadata differences
print("\n=== METADATA CHECK ===")
if 'metadata' in latest_signals.columns:
    # Get first metadata entry
    first_meta = latest_signals['metadata'].iloc[0]
    if isinstance(first_meta, dict):
        print("Risk parameters from metadata:")
        if 'risk' in first_meta:
            print(f"  Stop loss: {first_meta['risk'].get('stop_loss', 'N/A')}")
            print(f"  Take profit: {first_meta['risk'].get('take_profit', 'N/A')}")
            print(f"  Trailing stop: {first_meta['risk'].get('trailing_stop', 'N/A')}")
        
        print("\nStrategy parameters:")
        if 'parameters' in first_meta:
            params = first_meta['parameters']
            print(f"  Period: {params.get('period', 'N/A')}")
            print(f"  Std dev: {params.get('std_dev', 'N/A')}")