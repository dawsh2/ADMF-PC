#!/usr/bin/env python3
"""
Resample 1-minute SPY data to 15-minute bars.

This script:
1. Reads SPY_1m.parquet
2. Resamples to 15-minute OHLCV bars
3. Saves as SPY_15m.parquet and SPY_15m.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

def resample_to_15m(df_1m):
    """
    Resample 1-minute data to 15-minute bars.
    
    Proper OHLCV aggregation:
    - Open: first open price in the period
    - High: highest high in the period
    - Low: lowest low in the period
    - Close: last close price in the period
    - Volume: sum of volume in the period
    """
    # Ensure datetime index
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        if 'timestamp' in df_1m.columns:
            df_1m = df_1m.set_index('timestamp')
        elif 'datetime' in df_1m.columns:
            df_1m = df_1m.set_index('datetime')
        else:
            raise ValueError("No timestamp column found")
    
    # Sort by time to ensure correct aggregation
    df_1m = df_1m.sort_index()
    
    # Define aggregation rules for OHLCV
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Resample to 15-minute bars
    df_15m = df_1m.resample('15min').agg(agg_rules)
    
    # Remove any rows with NaN values (e.g., periods with no data)
    df_15m = df_15m.dropna()
    
    # Reset index to have timestamp as a column for CSV compatibility
    df_15m = df_15m.reset_index()
    df_15m = df_15m.rename(columns={'index': 'timestamp'})
    
    return df_15m

def main():
    # Define paths
    data_dir = Path('/Users/daws/ADMF-PC/data')
    input_file = data_dir / 'SPY_1m.parquet'
    output_parquet = data_dir / 'SPY_15m.parquet'
    output_csv = data_dir / 'SPY_15m.csv'
    
    print(f"Reading 1-minute data from {input_file}...")
    df_1m = pd.read_parquet(input_file)
    print(f"Loaded {len(df_1m)} 1-minute bars")
    print(f"Date range: {df_1m.index.min()} to {df_1m.index.max()}")
    
    print("\nResampling to 15-minute bars...")
    df_15m = resample_to_15m(df_1m)
    print(f"Created {len(df_15m)} 15-minute bars")
    print(f"Date range: {df_15m['timestamp'].min()} to {df_15m['timestamp'].max()}")
    
    # Show sample of the data
    print("\nSample of 15-minute data:")
    print(df_15m.head())
    
    # Save the resampled data
    print(f"\nSaving to {output_parquet}...")
    df_15m.to_parquet(output_parquet, index=False)
    
    print(f"Saving to {output_csv}...")
    df_15m.to_csv(output_csv, index=False)
    
    print("\nDone! Created:")
    print(f"  - {output_parquet}")
    print(f"  - {output_csv}")
    
    # Verify the files
    print("\nVerifying saved files...")
    df_verify = pd.read_parquet(output_parquet)
    print(f"Parquet file contains {len(df_verify)} rows")
    
    return df_15m

if __name__ == "__main__":
    df_15m = main()