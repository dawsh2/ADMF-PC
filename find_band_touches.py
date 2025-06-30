#!/usr/bin/env python3
"""Find where Bollinger Bands are actually touched in the data."""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/SPY_5m.csv')
print(f"Total data: {len(df)} bars")

# Calculate Bollinger Bands
period = 20
std_dev = 2.0

df['sma'] = df['close'].rolling(period).mean()
df['std'] = df['close'].rolling(period).std()
df['upper'] = df['sma'] + std_dev * df['std']
df['lower'] = df['sma'] - std_dev * df['std']

# Find touches
df['touches_upper'] = df['close'] >= df['upper']
df['touches_lower'] = df['close'] <= df['lower']

# Drop rows without bands
df = df.dropna()

# Count touches
upper_touches = df['touches_upper'].sum()
lower_touches = df['touches_lower'].sum()

print(f"\nTotal upper band touches: {upper_touches}")
print(f"Total lower band touches: {lower_touches}")

# Find first touches
first_upper = df[df['touches_upper']].index[0] if upper_touches > 0 else None
first_lower = df[df['touches_lower']].index[0] if lower_touches > 0 else None

print(f"\nFirst upper touch at bar: {first_upper}")
print(f"First lower touch at bar: {first_lower}")

# Check specific ranges
print("\n=== Touch analysis by data range ===")

# Training data (first 80%)
train_end = int(len(df) * 0.8)
train_df = df[:train_end]
print(f"\nTraining data (0-{train_end}):")
print(f"  Upper touches: {train_df['touches_upper'].sum()}")
print(f"  Lower touches: {train_df['touches_lower'].sum()}")

# Show some examples
if train_df['touches_upper'].sum() > 0:
    print("\n  First 5 upper touches in training:")
    upper_touches_train = train_df[train_df['touches_upper']]
    for idx in upper_touches_train.index[:5]:
        print(f"    Bar {idx}: close={df.loc[idx, 'close']:.2f}, upper={df.loc[idx, 'upper']:.2f}")

if train_df['touches_lower'].sum() > 0:
    print("\n  First 5 lower touches in training:")
    lower_touches_train = train_df[train_df['touches_lower']]
    for idx in lower_touches_train.index[:5]:
        print(f"    Bar {idx}: close={df.loc[idx, 'close']:.2f}, lower={df.loc[idx, 'lower']:.2f}")

# Check first 200 bars specifically
print(f"\n=== First 200 bars (what we're testing with) ===")
first_200 = df[:200]
print(f"Upper touches: {first_200['touches_upper'].sum()}")
print(f"Lower touches: {first_200['touches_lower'].sum()}")

if first_200['touches_upper'].sum() == 0 and first_200['touches_lower'].sum() == 0:
    print("\n⚠️  NO BAND TOUCHES IN FIRST 200 BARS!")
    print("This explains why we're getting 0 signals.")
    
    # Find closest approaches
    first_200['dist_to_upper'] = first_200['upper'] - first_200['close']
    first_200['dist_to_lower'] = first_200['close'] - first_200['lower']
    
    closest_upper_idx = first_200['dist_to_upper'].idxmin()
    closest_lower_idx = first_200['dist_to_lower'].idxmin()
    
    print(f"\nClosest to upper band:")
    print(f"  Bar {closest_upper_idx}: close={first_200.loc[closest_upper_idx, 'close']:.2f}, upper={first_200.loc[closest_upper_idx, 'upper']:.2f}, distance={first_200.loc[closest_upper_idx, 'dist_to_upper']:.4f}")
    
    print(f"\nClosest to lower band:")
    print(f"  Bar {closest_lower_idx}: close={first_200.loc[closest_lower_idx, 'close']:.2f}, lower={first_200.loc[closest_lower_idx, 'lower']:.2f}, distance={first_200.loc[closest_lower_idx, 'dist_to_lower']:.4f}")
    
    # Suggest a better range
    print("\n=== Suggested test ranges with signals ===")
    for start in [0, 500, 1000, 2000]:
        segment = df[start:start+200]
        u_touches = segment['touches_upper'].sum()
        l_touches = segment['touches_lower'].sum()
        if u_touches > 0 or l_touches > 0:
            print(f"  Bars {start}-{start+200}: {u_touches} upper, {l_touches} lower touches")