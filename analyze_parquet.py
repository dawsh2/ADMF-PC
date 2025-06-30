#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

# Read the parquet file
file_path = Path('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
if not file_path.exists():
    print(f"File not found: {file_path}")
    exit(1)

df = pd.read_parquet(file_path)

print('=== PARQUET FILE ANALYSIS ===')
print(f'\nDataFrame shape: {df.shape}')
print(f'Number of rows: {len(df)}')
print(f'Number of columns: {len(df.columns)}')

print('\n=== COLUMN INFORMATION ===')
print('\nColumn names:')
for col in df.columns:
    print(f'  - {col} ({df[col].dtype})')

print('\n=== SAMPLE DATA (first 5 rows) ===')
print(df.head())

print('\n=== SAMPLE DATA (last 5 rows) ===')
print(df.tail())

# Check for signal-related columns
signal_cols = [col for col in df.columns if 'signal' in col.lower()]
print(f'\n=== SIGNAL COLUMNS ===')
print(f'Found {len(signal_cols)} signal-related columns:')
for col in signal_cols:
    print(f'  - {col}')
    
# Analyze signal values
if 'signal' in df.columns:
    print('\n=== SIGNAL VALUE ANALYSIS ===')
    print(f'Unique signal values: {df["signal"].unique()}')
    print(f'Signal value counts:')
    print(df['signal'].value_counts())
    
    # Count signal changes
    signal_changes = (df['signal'] != df['signal'].shift()).sum() - 1  # -1 to exclude first row
    print(f'\nNumber of signal changes: {signal_changes}')

# Check for strategy-related columns
strategy_cols = [col for col in df.columns if 'strategy' in col.lower() or 'sub' in col.lower()]
print(f'\n=== STRATEGY-RELATED COLUMNS ===')
print(f'Found {len(strategy_cols)} strategy-related columns:')
for col in strategy_cols:
    print(f'  - {col}')
    if len(df[col].unique()) < 20:  # Only show unique values if reasonable number
        print(f'    Unique values: {df[col].unique()}')

# Check for metadata columns
metadata_cols = [col for col in df.columns if 'meta' in col.lower() or 'source' in col.lower() or 'component' in col.lower()]
print(f'\n=== METADATA COLUMNS ===')
print(f'Found {len(metadata_cols)} metadata columns:')
for col in metadata_cols:
    print(f'  - {col}')
    if len(df[col].unique()) < 20:
        print(f'    Unique values: {df[col].unique()}')

# Additional analysis for signals
print('\n=== SIGNAL TRANSITION ANALYSIS ===')
if 'signal' in df.columns:
    # Find where signals change
    signal_changes_mask = df['signal'] != df['signal'].shift()
    signal_change_points = df[signal_changes_mask]
    
    print(f'Total signal changes: {len(signal_change_points) - 1}')  # -1 for first row
    print('\nFirst 10 signal changes:')
    print(signal_change_points.head(10)[['timestamp', 'signal'] if 'timestamp' in df.columns else ['signal']])

# Check if there's information about which strategy generated signals
print('\n=== LOOKING FOR SUB-STRATEGY INFORMATION ===')
for col in df.columns:
    if 'strategy' in col.lower() or 'source' in col.lower() or 'component' in col.lower():
        unique_vals = df[col].unique()
        if len(unique_vals) > 1 and len(unique_vals) < 50:
            print(f'\n{col} has {len(unique_vals)} unique values:')
            print(unique_vals[:20])  # Show first 20