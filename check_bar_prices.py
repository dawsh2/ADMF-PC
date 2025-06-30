#!/usr/bin/env python3
"""Check if BAR data has correct high/low prices."""

import pandas as pd
from pathlib import Path

print("=== Checking BAR Price Data ===")

# Load raw data
data_file = Path("data/spy_5m_full.parquet")
if data_file.exists():
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df)} bars")
    
    # Check a few samples
    print("\nSample bars:")
    sample = df.head(5)
    for i, row in sample.iterrows():
        print(f"\n{row['timestamp']}")
        print(f"  Open:  {row['open']:.4f}")
        print(f"  High:  {row['high']:.4f}")
        print(f"  Low:   {row['low']:.4f}")
        print(f"  Close: {row['close']:.4f}")
        
        # Sanity checks
        if row['high'] < row['low']:
            print("  ⚠️ ERROR: High < Low!")
        if row['high'] < row['open'] or row['high'] < row['close']:
            print("  ⚠️ ERROR: High is not the highest price!")
        if row['low'] > row['open'] or row['low'] > row['close']:
            print("  ⚠️ ERROR: Low is not the lowest price!")
    
    # Check for any bars with inverted high/low
    inverted = df[df['high'] < df['low']]
    print(f"\n\nBars with high < low: {len(inverted)}")
    
    # Check typical spreads
    df['spread_pct'] = ((df['high'] - df['low']) / df['low']) * 100
    print(f"\nAverage high-low spread: {df['spread_pct'].mean():.4f}%")
    print(f"Median high-low spread: {df['spread_pct'].median():.4f}%")
    print(f"Max high-low spread: {df['spread_pct'].max():.4f}%")
    
    # Check if 0.075% moves are common within bars
    df['move_from_open'] = abs((df['close'] - df['open']) / df['open']) * 100
    moves_above_threshold = df[df['move_from_open'] > 0.075]
    print(f"\nBars with >0.075% move from open: {len(moves_above_threshold)} ({len(moves_above_threshold)/len(df)*100:.1f}%)")
    
    # Check if high-low range often exceeds stop loss
    df['range_pct'] = ((df['high'] - df['low']) / df['low']) * 100
    ranges_above_sl = df[df['range_pct'] > 0.075]
    print(f"Bars with range >0.075%: {len(ranges_above_sl)} ({len(ranges_above_sl)/len(df)*100:.1f}%)")
    
    # This tells us how often a stop loss COULD be hit within a single bar
    ranges_above_both = df[df['range_pct'] > 0.175]  # 0.075% + 0.1%
    print(f"Bars with range >0.175% (SL+TP): {len(ranges_above_both)} ({len(ranges_above_both)/len(df)*100:.1f}%)")
    
else:
    print(f"Data file not found: {data_file}")

# Also check for the specific issue - positions exiting at gain when should be loss
print("\n\n=== Checking Stop Loss Logic ===")
print("Stop loss is configured as 0.075%")
print("For a LONG position:")
print("  - Entry at $100.00")
print("  - Stop loss should trigger at $99.925 (-0.075%)")
print("  - NOT at $100.075 (+0.075%)")
print("\nIf you're seeing exits at +0.075%, the calculation is inverted somewhere.")