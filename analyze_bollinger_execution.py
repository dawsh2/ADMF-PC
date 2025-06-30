#!/usr/bin/env python3
"""Analyze execution details for Bollinger strategy with stops/targets"""

import pandas as pd
import numpy as np
import json

# Load fills data
fills = pd.read_parquet('config/bollinger/results/latest/traces/execution/fills/execution_fills.parquet')
positions = pd.read_parquet('config/bollinger/results/latest/traces/portfolio/positions_close/positions_close.parquet')

print("FILL ANALYSIS")
print("="*60)
print(f"Total fills: {len(fills)}")

# Extract fill metadata
fill_data = []
for idx, row in fills.iterrows():
    meta = json.loads(row['metadata'])
    fill_data.append({
        'timestamp': row['ts'],
        'side': meta['side'],
        'price': meta.get('price', meta.get('fill_price')),
        'quantity': meta['quantity'],
        'commission': meta['commission']
    })

fills_df = pd.DataFrame(fill_data)
print(f"\nFill sides:")
print(fills_df['side'].value_counts())

print("\n" + "="*60)
print("POSITION ANALYSIS")
print("="*60)
print(f"Total positions: {len(positions)}")

# Extract position metadata
position_data = []
for idx, row in positions.iterrows():
    meta = json.loads(row['metadata'])
    exit_reason = meta.get('exit_reason', 'unknown')
    
    # Check if position hit stop or target
    if 'stop_loss' in exit_reason.lower():
        exit_type = 'stop_loss'
    elif 'take_profit' in exit_reason.lower() or 'target' in exit_reason.lower():
        exit_type = 'take_profit'
    elif 'eod' in exit_reason.lower() or 'end of day' in exit_reason.lower():
        exit_type = 'eod_exit'
    else:
        exit_type = 'signal_exit'
    
    position_data.append({
        'entry_price': float(row['entry_price']),
        'exit_price': float(row['exit_price']),
        'realized_pnl': float(meta['realized_pnl']),
        'exit_type': exit_type,
        'exit_reason': exit_reason,
        'return_pct': (float(row['exit_price']) - float(row['entry_price'])) / float(row['entry_price']) * 100
    })

positions_df = pd.DataFrame(position_data)

print("\nExit types:")
print(positions_df['exit_type'].value_counts())

print("\nReturns by exit type:")
for exit_type in positions_df['exit_type'].unique():
    subset = positions_df[positions_df['exit_type'] == exit_type]
    print(f"\n{exit_type}:")
    print(f"  Count: {len(subset)}")
    print(f"  Avg return: {subset['return_pct'].mean():.3f}%")
    print(f"  Win rate: {(subset['return_pct'] > 0).mean()*100:.1f}%")

print("\n" + "="*60)
print("STOP/TARGET ANALYSIS")
print("="*60)

# Configured stop/target
stop_pct = 0.1  # 0.1% stop loss
target_pct = 0.15  # 0.15% take profit

# Check if returns match expected stops/targets
stop_hits = positions_df[positions_df['exit_type'] == 'stop_loss']
if len(stop_hits) > 0:
    print(f"\nStop loss hits: {len(stop_hits)}")
    print(f"Average stop loss return: {stop_hits['return_pct'].mean():.3f}%")
    print(f"Expected stop loss: -{stop_pct}%")

target_hits = positions_df[positions_df['exit_type'] == 'take_profit']
if len(target_hits) > 0:
    print(f"\nTake profit hits: {len(target_hits)}")
    print(f"Average take profit return: {target_hits['return_pct'].mean():.3f}%")
    print(f"Expected take profit: {target_pct}%")

# Look for positions that SHOULD have hit stops/targets
print("\n" + "="*60)
print("POSITIONS THAT SHOULD HAVE HIT STOPS/TARGETS")
print("="*60)

missed_stops = positions_df[(positions_df['return_pct'] < -stop_pct) & (positions_df['exit_type'] != 'stop_loss')]
if len(missed_stops) > 0:
    print(f"\nPositions that went below -{stop_pct}% but didn't stop out: {len(missed_stops)}")
    print("Sample missed stops:")
    print(missed_stops[['return_pct', 'exit_type', 'exit_reason']].head())

missed_targets = positions_df[(positions_df['return_pct'] > target_pct) & (positions_df['exit_type'] != 'take_profit')]
if len(missed_targets) > 0:
    print(f"\nPositions that exceeded {target_pct}% but didn't take profit: {len(missed_targets)}")
    print("Sample missed targets:")
    print(missed_targets[['return_pct', 'exit_type', 'exit_reason']].head())

# Overall statistics
print("\n" + "="*60)
print("OVERALL PERFORMANCE")
print("="*60)
print(f"Total return: {positions_df['realized_pnl'].sum():.2f}")
print(f"Average return per trade: {positions_df['return_pct'].mean():.3f}%")
print(f"Win rate: {(positions_df['return_pct'] > 0).mean()*100:.1f}%")
print(f"Largest win: {positions_df['return_pct'].max():.3f}%")
print(f"Largest loss: {positions_df['return_pct'].min():.3f}%")

# Distribution of returns
print("\nReturn distribution:")
print(f"  Returns < -0.1%: {(positions_df['return_pct'] < -0.1).sum()}")
print(f"  Returns -0.1% to 0%: {((positions_df['return_pct'] >= -0.1) & (positions_df['return_pct'] < 0)).sum()}")
print(f"  Returns 0% to 0.15%: {((positions_df['return_pct'] >= 0) & (positions_df['return_pct'] <= 0.15)).sum()}")
print(f"  Returns > 0.15%: {(positions_df['return_pct'] > 0.15).sum()}")