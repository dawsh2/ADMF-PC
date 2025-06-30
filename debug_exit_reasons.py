#!/usr/bin/env python3
"""Debug why all exits show as take profit hits"""

import pandas as pd
import json

# Load positions
positions = pd.read_parquet('config/bollinger/results/latest/traces/portfolio/positions_close/positions_close.parquet')

print("ANALYZING EXIT REASONS")
print("="*60)

# Sample some positions with different returns
sample_positions = []
for idx, row in positions.iterrows():
    meta = json.loads(row['metadata'])
    return_pct = (float(row['exit_price']) - float(row['entry_price'])) / float(row['entry_price']) * 100
    
    sample_positions.append({
        'idx': idx,
        'entry_price': float(row['entry_price']),
        'exit_price': float(row['exit_price']),
        'return_pct': return_pct,
        'exit_reason': meta.get('exit_reason', 'unknown'),
        'exit_type': row.get('exit_type', 'unknown')
    })

df = pd.DataFrame(sample_positions)

# Group by return ranges
print("\nPositions by return range:")
print("-0.15%:", len(df[df['return_pct'] == -0.15]))
print("-0.15% to 0%:", len(df[(df['return_pct'] > -0.15) & (df['return_pct'] < 0)]))
print("0% to 0.15%:", len(df[(df['return_pct'] > 0) & (df['return_pct'] < 0.15)]))
print("0.15%:", len(df[df['return_pct'] == 0.15]))

# Check exit reasons for different return levels
print("\n" + "="*60)
print("EXIT REASONS BY RETURN LEVEL")
print("="*60)

# Exactly -0.15% (should be stop losses)
stops = df[df['return_pct'] == -0.15]
if len(stops) > 0:
    print(f"\nPositions at exactly -0.15% ({len(stops)} total):")
    print("Exit reasons:")
    print(stops['exit_reason'].value_counts())
    print("\nSample:")
    print(stops[['entry_price', 'exit_price', 'exit_reason']].head())

# Exactly 0.15% (should be take profits)
targets = df[df['return_pct'] == 0.15]
if len(targets) > 0:
    print(f"\nPositions at exactly 0.15% ({len(targets)} total):")
    print("Exit reasons:")
    print(targets['exit_reason'].value_counts())
    print("\nSample:")
    print(targets[['entry_price', 'exit_price', 'exit_reason']].head())

# Other returns (should be signal exits)
others = df[(df['return_pct'] != -0.15) & (df['return_pct'] != 0.15)]
if len(others) > 0:
    print(f"\nPositions at other return levels ({len(others)} total):")
    print("Exit reasons:")
    print(others['exit_reason'].value_counts())
    print("\nReturn distribution:")
    print(f"  Min: {others['return_pct'].min():.3f}%")
    print(f"  Max: {others['return_pct'].max():.3f}%")
    print(f"  Mean: {others['return_pct'].mean():.3f}%")

# Check for wrong exit price calculations
print("\n" + "="*60)
print("CHECKING EXIT PRICE CALCULATIONS")
print("="*60)

# For -0.15% returns, check if exit price = entry * 0.9985
stops_check = df[df['return_pct'] == -0.15].copy()
stops_check['expected_exit'] = stops_check['entry_price'] * 0.9985
stops_check['price_diff'] = abs(stops_check['exit_price'] - stops_check['expected_exit'])

print("\nStop loss price accuracy:")
print(f"Average price difference: ${stops_check['price_diff'].mean():.6f}")
print(f"Max price difference: ${stops_check['price_diff'].max():.6f}")

# For 0.15% returns, check if exit price = entry * 1.0015
targets_check = df[df['return_pct'] == 0.15].copy()
targets_check['expected_exit'] = targets_check['entry_price'] * 1.0015
targets_check['price_diff'] = abs(targets_check['exit_price'] - targets_check['expected_exit'])

print("\nTake profit price accuracy:")
print(f"Average price difference: ${targets_check['price_diff'].mean():.6f}")
print(f"Max price difference: ${targets_check['price_diff'].max():.6f}")