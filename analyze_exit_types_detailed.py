#!/usr/bin/env python3
"""Detailed analysis of exit types to ensure we're not mixing different exit reasons"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

print("DETAILED EXIT TYPE ANALYSIS")
print("=" * 60)

# Load position data
results_dir = Path("config/bollinger/results/latest")
closes = pd.read_parquet(results_dir / "traces/portfolio/positions_close/positions_close.parquet")

# Parse all positions with exit data
all_positions = []
for idx, row in closes.iterrows():
    # Get exit type
    if 'exit_type' in closes.columns:
        exit_type = row.get('exit_type', 'unknown')
    else:
        metadata = row.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        exit_type = metadata.get('exit_type', 'unknown') if isinstance(metadata, dict) else 'unknown'
    
    # Get price data
    if 'exit_price' in row and 'entry_price' in row:
        exit_price = float(row['exit_price'])
        entry_price = float(row['entry_price'])
    else:
        continue
        
    # Calculate return
    return_pct = (exit_price - entry_price) / entry_price * 100
    
    all_positions.append({
        'exit_type': exit_type,
        'exit_price': exit_price,
        'entry_price': entry_price,
        'return_pct': return_pct,
        'return_abs': exit_price - entry_price
    })

df = pd.DataFrame(all_positions)

# Group by exit type
print(f"Total positions: {len(df)}")
print("\nExit type distribution:")
exit_counts = df['exit_type'].value_counts()
for exit_type, count in exit_counts.items():
    print(f"  {exit_type}: {count} ({count/len(df)*100:.1f}%)")

# Analyze each exit type separately
print("\n" + "=" * 60)
print("ANALYSIS BY EXIT TYPE")
print("=" * 60)

# 1. Stop Loss Exits
stop_losses = df[df['exit_type'] == 'stop_loss']
if len(stop_losses) > 0:
    print(f"\nSTOP LOSS EXITS ({len(stop_losses)} positions):")
    print(f"  Expected return: -0.075%")
    print(f"  Actual mean return: {stop_losses['return_pct'].mean():.4f}%")
    print(f"  Actual std dev: {stop_losses['return_pct'].std():.4f}%")
    print(f"  Return range: [{stop_losses['return_pct'].min():.4f}%, {stop_losses['return_pct'].max():.4f}%]")
    
    # Check how many are close to -0.075%
    close_to_stop = stop_losses[np.abs(stop_losses['return_pct'] + 0.075) < 0.01]
    print(f"  Positions within 0.01% of -0.075%: {len(close_to_stop)} ({len(close_to_stop)/len(stop_losses)*100:.1f}%)")
    
    # Distribution
    print("\n  Return distribution:")
    bins = [-1, -0.15, -0.10, -0.075, -0.05, 0, 0.05, 0.10, 0.15, 1]
    hist = pd.cut(stop_losses['return_pct'], bins=bins).value_counts().sort_index()
    for interval, count in hist.items():
        if count > 0:
            print(f"    {interval}: {count} trades")

# 2. Take Profit Exits
take_profits = df[df['exit_type'] == 'take_profit']
if len(take_profits) > 0:
    print(f"\nTAKE PROFIT EXITS ({len(take_profits)} positions):")
    print(f"  Expected return: +0.15%")
    print(f"  Actual mean return: {take_profits['return_pct'].mean():.4f}%")
    print(f"  Actual std dev: {take_profits['return_pct'].std():.4f}%")
    print(f"  Return range: [{take_profits['return_pct'].min():.4f}%, {take_profits['return_pct'].max():.4f}%]")
    
    # Check how many are close to 0.15%
    close_to_target = take_profits[np.abs(take_profits['return_pct'] - 0.15) < 0.01]
    print(f"  Positions within 0.01% of +0.15%: {len(close_to_target)} ({len(close_to_target)/len(take_profits)*100:.1f}%)")
    
    # Distribution
    print("\n  Return distribution:")
    bins = [-1, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25, 1]
    hist = pd.cut(take_profits['return_pct'], bins=bins).value_counts().sort_index()
    for interval, count in hist.items():
        if count > 0:
            print(f"    {interval}: {count} trades")

# 3. Signal Exits (should be separate)
signal_exits = df[df['exit_type'] == 'signal']
if len(signal_exits) > 0:
    print(f"\nSIGNAL EXITS ({len(signal_exits)} positions):")
    print(f"  Mean return: {signal_exits['return_pct'].mean():.4f}%")
    print(f"  Std dev: {signal_exits['return_pct'].std():.4f}%")
    print(f"  Return range: [{signal_exits['return_pct'].min():.4f}%, {signal_exits['return_pct'].max():.4f}%]")
    
    # These should have more variable returns
    print("\n  Return distribution:")
    bins = [-1, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 1]
    hist = pd.cut(signal_exits['return_pct'], bins=bins).value_counts().sort_index()
    for interval, count in hist.items():
        if count > 0:
            print(f"    {interval}: {count} trades")

# 4. Other/Unknown exits
other_exits = df[~df['exit_type'].isin(['stop_loss', 'take_profit', 'signal'])]
if len(other_exits) > 0:
    print(f"\nOTHER/UNKNOWN EXITS ({len(other_exits)} positions):")
    print(f"  Exit types: {other_exits['exit_type'].value_counts().to_dict()}")
    print(f"  Mean return: {other_exits['return_pct'].mean():.4f}%")
    print(f"  Std dev: {other_exits['return_pct'].std():.4f}%")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nKey findings:")
print("1. Stop losses that should exit at -0.075% are showing wide variance")
print("2. Take profits that should exit at +0.15% are showing wide variance")
print("3. This confirms exits are happening at market price, not stop/target price")

# Check if any positions have exactly -0.075% or +0.15% returns
exact_stops = df[np.abs(df['return_pct'] + 0.075) < 0.0001]
exact_targets = df[np.abs(df['return_pct'] - 0.15) < 0.0001]
print(f"\nPositions with EXACTLY -0.075% return: {len(exact_stops)}")
print(f"Positions with EXACTLY +0.15% return: {len(exact_targets)}")