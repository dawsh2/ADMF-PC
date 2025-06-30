#!/usr/bin/env python3
"""Check data alignment between signals and source data."""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

# Load a signal file
workspace = "workspaces/signal_generation_a2d31737"
signal_files = sorted(glob(str(Path(workspace) / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones/*.parquet")))
signals_df = pd.read_parquet(signal_files[0])

print("SIGNAL DATA:")
print(f"Shape: {signals_df.shape}")
print(f"Date range: {signals_df['ts'].min()} to {signals_df['ts'].max()}")
print(f"First few timestamps:")
for ts in signals_df['ts'].head(10):
    print(f"  {ts}")

# Check available data files
print("\n\nAVAILABLE DATA FILES:")
for data_file in ["./data/SPY_5m.csv", "./data/SPY_1m.csv"]:
    if Path(data_file).exists():
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"\n{data_file}:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Check if signal dates are in data range
        signal_start = pd.to_datetime(signals_df['ts'].min())
        signal_end = pd.to_datetime(signals_df['ts'].max())
        
        data_contains_signals = (signal_start >= df['timestamp'].min() and 
                                signal_end <= df['timestamp'].max())
        
        print(f"  Contains signal dates: {data_contains_signals}")
        
        if data_contains_signals:
            # Filter to signal date range
            filtered = df[(df['timestamp'] >= signal_start) & (df['timestamp'] <= signal_end)]
            print(f"  Bars in signal date range: {len(filtered)}")
            
            # Sample some calculations
            filtered['sma_20'] = filtered['close'].rolling(20).mean()
            filtered['atr'] = filtered['high'] - filtered['low']
            filtered['atr_20'] = filtered['atr'].rolling(20).mean()
            
            print(f"  Non-null SMA values: {filtered['sma_20'].notna().sum()}")
            print(f"  Non-null ATR values: {filtered['atr_20'].notna().sum()}")

# Check what the issue might be
print("\n\nDEBUGGING ENHANCED SIGNALS:")
# Try to create enhanced signals
signals_df['timestamp'] = pd.to_datetime(signals_df['ts'])

# Use 5m data since that exists
if Path("./data/SPY_5m.csv").exists():
    source_df = pd.read_csv("./data/SPY_5m.csv")
    source_df['timestamp'] = pd.to_datetime(source_df['timestamp'])
    
    # Filter to signal range
    start = signals_df['timestamp'].min()
    end = signals_df['timestamp'].max()
    source_df = source_df[(source_df['timestamp'] >= start) & (source_df['timestamp'] <= end)]
    
    print(f"Source data in range: {len(source_df)} bars")
    
    # Try simple merge
    merged = pd.merge_asof(
        signals_df[['timestamp', 'val', 'px']].sort_values('timestamp'),
        source_df[['timestamp', 'close', 'volume']].sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    
    print(f"Merged shape: {merged.shape}")
    print(f"Non-null close values: {merged['close'].notna().sum()}")
    print("\nFirst few merged rows:")
    print(merged.head(10))