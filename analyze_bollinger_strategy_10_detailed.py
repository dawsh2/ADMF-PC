#!/usr/bin/env python3
"""Analyze the bollinger strategy #10 from the parent test in detail"""

import pandas as pd
import numpy as np
from datetime import datetime

# Read the strategy #10 trace file from parent test
trace_file = "config/bollinger/results/20250623_062931/traces/bollinger_bands/SPY_5m_compiled_strategy_10.parquet"

print("=== Bollinger Strategy #10 Detailed Analysis ===")
print(f"Parameters: period=11, std_dev=2.0")
print()

try:
    df = pd.read_parquet(trace_file)
    print(f"Total trace records: {len(df):,}")
    
    # Show columns
    print(f"\nColumns in trace file: {list(df.columns)}")
    
    # Check for signal changes
    if 'signal' in df.columns:
        signal_changes = df['signal'].diff().fillna(0) != 0
        num_signal_changes = signal_changes.sum()
        print(f"\nSignal changes: {num_signal_changes}")
        
        # Count buy/sell signals
        buy_signals = (df['signal'] > 0).sum()
        sell_signals = (df['signal'] < 0).sum()
        neutral_signals = (df['signal'] == 0).sum()
        
        print(f"Buy signals (signal > 0): {buy_signals:,}")
        print(f"Sell signals (signal < 0): {sell_signals:,}")
        print(f"Neutral signals (signal == 0): {neutral_signals:,}")
        
        # Show first few signal changes
        if num_signal_changes > 0:
            print("\nFirst 10 signal changes:")
            signal_change_df = df[signal_changes].head(10)
            for idx, row in signal_change_df.iterrows():
                print(f"  Index {idx}: signal = {row.get('signal', 'N/A')}")
    
    # Check date range
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    elif 'index' in df.columns:
        print(f"\nIndex range: {df['index'].min()} to {df['index'].max()}")
    
    # Sample of data
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nLast 5 rows:")
    print(df.tail())
    
except Exception as e:
    print(f"Error reading parquet file: {e}")
    print("The file might be using a different format or schema")

# Compare with bollinger_10 test
print("\n=== Comparison with bollinger_10 isolated test ===")
print("Parent test (large sweep): Generated 4,698 trace records")
print("Isolated bollinger_10 test: Generated 0 signals")
print()
print("This suggests:")
print("1. The parent test might be using different data or a different time period")
print("2. The isolated test might have stricter signal generation criteria")
print("3. The configuration might be slightly different between the two tests")