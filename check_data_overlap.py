#!/usr/bin/env python3
"""Check data overlap between notebook and latest results."""

import pandas as pd
from pathlib import Path

# Load both datasets
notebook_signals = pd.read_parquet("config/bollinger/results/20250625_170201/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
latest_signals = pd.read_parquet("config/bollinger/results/latest/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")

notebook_signals['ts'] = pd.to_datetime(notebook_signals['ts'])
latest_signals['ts'] = pd.to_datetime(latest_signals['ts'])

print("=== Time Period Analysis ===")
print(f"Notebook period: {notebook_signals['ts'].min()} to {notebook_signals['ts'].max()}")
print(f"Latest period: {latest_signals['ts'].min()} to {latest_signals['ts'].max()}")

# Check overlap
notebook_start = notebook_signals['ts'].min()
notebook_end = notebook_signals['ts'].max()
latest_start = latest_signals['ts'].min()
latest_end = latest_signals['ts'].max()

print(f"\nOverlap check:")
print(f"Notebook starts at: {notebook_start}")
print(f"Latest ends at: {latest_end}")

if notebook_start > latest_end:
    print("❌ NO OVERLAP - Notebook data is AFTER latest results!")
    print("The notebook analyzed future data that your current system hasn't seen yet.")
else:
    print("✅ There is overlap")

# Calculate trading days
notebook_days = (notebook_end - notebook_start).days
latest_days = (latest_end - latest_start).days

print(f"\nTrading period duration:")
print(f"Notebook: {notebook_days} days")
print(f"Latest: {latest_days} days")

# Calculate trades per day
notebook_entries = 416  # From previous analysis
latest_entries = 1458   # From your results

print(f"\nTrades per day:")
print(f"Notebook: {notebook_entries / notebook_days:.2f} trades/day")
print(f"Latest: {latest_entries / latest_days:.2f} trades/day")

# Load market data to understand full dataset
market_data = pd.read_csv("config/bollinger/results/20250625_170201/data/SPY_5m.csv")
market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])

print(f"\n=== Full Market Data ===")
print(f"Market data period: {market_data['timestamp'].min()} to {market_data['timestamp'].max()}")
print(f"Total bars: {len(market_data)}")

# The notebook likely split data for train/test
total_days = (market_data['timestamp'].max() - market_data['timestamp'].min()).days
print(f"Total days in dataset: {total_days}")

# Check if notebook used last ~2 months as test set
test_start = pd.Timestamp('2025-01-28')
test_data = market_data[market_data['timestamp'] >= test_start]
print(f"\nData from Jan 28, 2025 onwards:")
print(f"Bars: {len(test_data)} ({len(test_data)/len(market_data)*100:.1f}% of total)")
print(f"This appears to be the TEST SET used in the notebook")