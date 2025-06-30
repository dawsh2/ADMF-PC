#!/usr/bin/env python3
"""Check the metadata format in signals and positions."""

import pandas as pd
import json
from pathlib import Path

print("=== Checking Metadata Format ===")

results_dir = Path("config/bollinger/results/latest")

# Check signals first
signals_file = results_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"
if signals_file.exists():
    signals = pd.read_parquet(signals_file)
    print(f"\nSignals file: {len(signals)} rows")
    
    # Get first signal
    if len(signals) > 0:
        first = signals.iloc[0]
        print("\nFirst signal structure:")
        for col in signals.columns:
            val = first[col]
            print(f"  {col}: {type(val).__name__} = {str(val)[:100]}...")
        
        # Try to parse metadata if it's a string
        if 'metadata' in signals.columns and isinstance(first['metadata'], str):
            try:
                metadata_dict = json.loads(first['metadata'])
                print("\n\nParsed metadata:")
                for k, v in metadata_dict.items():
                    if k in ['open', 'high', 'low', 'close', 'price']:
                        print(f"  {k}: {v}")
                
                # Check if we added OHLC
                has_ohlc = all(k in metadata_dict for k in ['open', 'high', 'low', 'close'])
                print(f"\nHas OHLC data: {has_ohlc}")
                
            except Exception as e:
                print(f"\nError parsing metadata: {e}")

# Check position closes
pos_close_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"
if pos_close_file.exists():
    closes = pd.read_parquet(pos_close_file)
    print(f"\n\nPosition closes: {len(closes)} rows")
    print("\nColumns:", list(closes.columns))
    
    # Check for strategy_id in various places
    if len(closes) > 0:
        first_close = closes.iloc[0]
        
        # Direct column
        if 'strategy_id' in closes.columns:
            print(f"\n✓ strategy_id column exists")
            print(f"  Non-null values: {closes['strategy_id'].notna().sum()}")
        else:
            print("\n❌ No strategy_id column")
        
        # In metadata
        if 'metadata' in closes.columns:
            if isinstance(first_close['metadata'], str):
                try:
                    meta = json.loads(first_close['metadata'])
                    if 'strategy_id' in meta:
                        print(f"✓ strategy_id found in metadata: {meta['strategy_id']}")
                except:
                    pass
            elif isinstance(first_close['metadata'], dict):
                if 'strategy_id' in first_close['metadata']:
                    print(f"✓ strategy_id found in metadata: {first_close['metadata']['strategy_id']}")

print("\n\n=== Issue Summary ===")
print("1. Metadata is stored as JSON strings, not dicts")
print("2. strategy_id is missing from position events")
print("3. Need to check if OHLC data was actually added to signals")