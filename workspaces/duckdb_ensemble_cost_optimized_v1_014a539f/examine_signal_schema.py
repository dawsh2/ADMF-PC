"""
Examine the actual schema of the signal trace parquet file.
"""
import pandas as pd
import numpy as np

# Load the sparse signal trace parquet file
parquet_path = 'traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_cost_optimized.parquet'
print(f"Loading signal traces from: {parquet_path}")
df = pd.read_parquet(parquet_path)

print(f"\nDataFrame Info:")
print(df.info())

print(f"\nDetailed column analysis:")
for col in df.columns:
    print(f"\n{col}:")
    print(f"  Type: {df[col].dtype}")
    print(f"  Unique values: {df[col].nunique()}")
    if df[col].dtype in ['int64', 'float64']:
        print(f"  Min: {df[col].min()}")
        print(f"  Max: {df[col].max()}")
        print(f"  Mean: {df[col].mean():.4f}")
    print(f"  Sample values: {df[col].head(3).tolist()}")

# Check if 'val' is the signal and 'px' is the price
print(f"\n\nAnalyzing 'val' as potential signal column:")
print(f"Unique values in 'val': {sorted(df['val'].unique())}")

print(f"\n\nAnalyzing 'px' as potential price column:")
print(f"Price range: {df['px'].min():.2f} to {df['px'].max():.2f}")

# Check the strategy name
print(f"\n\nStrategy name: {df['strat'].unique()}")

# Convert timestamp and sort
df['ts'] = pd.to_datetime(df['ts'])
df = df.sort_values('ts').reset_index(drop=True)

# Calculate time differences to understand sparse storage
df['time_diff'] = df['ts'].diff()
print(f"\n\nTime difference analysis (sparse storage):")
print(f"Average time between signal changes: {df['time_diff'].mean()}")
print(f"Max time between signal changes: {df['time_diff'].max()}")
print(f"Min time between signal changes: {df['time_diff'].min()}")

# Show transitions in signal values
print(f"\n\nSignal transitions (first 20):")
print(df[['idx', 'ts', 'val', 'px']].head(20))

# Check for price changes when signal changes
df['val_change'] = df['val'].diff()
df['px_change'] = df['px'].diff()
signal_changes = df[df['val_change'] != 0]
print(f"\n\nSignal changes analysis:")
print(f"Total signal changes: {len(signal_changes)}")
print(f"\nFirst 10 signal changes:")
print(signal_changes[['idx', 'ts', 'val', 'px', 'val_change', 'px_change']].head(10))