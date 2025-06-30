#!/usr/bin/env python3
"""Check for symbol mismatch between BAR events and positions."""

import pandas as pd
import json
from pathlib import Path

workspace = Path("config/bollinger/results/latest")

# Check position symbols
positions_open = pd.read_parquet(workspace / "traces/portfolio/positions_open/positions_open.parquet")
print("=== Position Symbols ===")
for _, row in positions_open.iterrows():
    metadata = json.loads(row['metadata'])
    print(f"Position symbol: {metadata['symbol']}")
    print(f"Raw sym column: {row['sym']}")

# Check what symbols are in the data
data_path = Path("./data/SPY_5m.csv")
if data_path.exists():
    bars = pd.read_csv(data_path, nrows=5)
    print("\n=== Bar Data Symbols ===")
    if 'symbol' in bars.columns:
        print(f"Symbol column values: {bars['symbol'].unique()}")
    else:
        print("No symbol column in bar data")
        print(f"Columns: {list(bars.columns)}")

# Check metadata for symbol mapping
with open(workspace / "metadata.json") as f:
    metadata = json.load(f)
    
print("\n=== Configuration Symbols ===")
print(f"Data symbols: {metadata['data_source']['symbols']}")
print(f"Component keys: {list(metadata['components'].keys())}")