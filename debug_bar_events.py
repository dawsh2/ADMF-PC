#!/usr/bin/env python3
"""Debug why BAR events aren't reaching portfolio for stop loss checking."""

import pandas as pd
import json
from pathlib import Path

workspace = Path("config/bollinger/results/latest")

# Check signal symbols
signals = pd.read_parquet(workspace / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
print("=== Signal Symbols ===")
print(f"Unique symbols in signals: {signals['sym'].unique()}")

# Check position symbols
positions_open = pd.read_parquet(workspace / "traces/portfolio/positions_open/positions_open.parquet")
print("\n=== Position Symbols ===")
print(f"Symbols in positions_open table: {positions_open['sym'].unique()}")
for _, row in positions_open.iterrows():
    metadata = json.loads(row['metadata'])
    print(f"Position metadata symbol: {metadata['symbol']}")

# Check order symbols
orders = pd.read_parquet(workspace / "traces/portfolio/orders/portfolio_orders.parquet")
print("\n=== Order Symbols ===")
print(f"Symbols in orders table: {orders['sym'].unique()}")

# Check the actual data file to see what symbol would be in BAR events
data_path = Path("./data/SPY_5m.csv")
if data_path.exists():
    bars = pd.read_csv(data_path, nrows=5)
    print("\n=== Data File Info ===")
    print(f"Data file: {data_path}")
    print(f"Columns: {list(bars.columns)}")
    if 'symbol' in bars.columns:
        print(f"Symbol in data: {bars['symbol'].iloc[0]}")
    else:
        print("No symbol column in data - would use filename: SPY_5m")

# Check metadata for configured symbols
with open(workspace / "metadata.json") as f:
    metadata = json.load(f)
print("\n=== Configuration ===")
print(f"Configured symbols: {metadata['data_source']['symbols']}")
print(f"Configured timeframes: {metadata['data_source']['timeframes']}")

print("\n=== Analysis ===")
print("BAR events would have symbol: SPY_5m (from filename)")
print("Portfolio positions have symbol: SPY_5m")
print("So symbol matching should work...")
print("\nThe issue must be that portfolio isn't subscribed to BAR events!")