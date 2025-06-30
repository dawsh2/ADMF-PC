#!/usr/bin/env python3
"""Check what data split is being used."""

import pandas as pd
from pathlib import Path

# Load market data
market_data = pd.read_parquet('data/SPY_5m.parquet')
signals = pd.read_parquet('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')

print("Data Split Analysis")
print("=" * 50)

print(f"\nMarket data:")
print(f"  Total bars: {len(market_data):,}")
print(f"  Date range: {market_data.index[0]} to {market_data.index[-1]}")

print(f"\nSignals data:")
print(f"  Max bar index: {signals['idx'].max():,}")
print(f"  Signal count: {len(signals):,}")

# Check source file reference
if 'src_file' in signals.columns:
    src_file = signals['src_file'].iloc[0] if len(signals) > 0 else 'Unknown'
    print(f"  Source file: {src_file}")
    
    # This tells us which data split was used
    if '1m' in str(src_file):
        print("  ⚠️  Using 1-minute data file, not 5-minute!")
        print("     Your previous results might have been on different data")

# Calculate what portion of data was used
bars_used = signals['idx'].max() + 1
data_fraction = bars_used / len(market_data)

print(f"\nData usage:")
print(f"  Bars used: {bars_used:,} out of {len(market_data):,}")
print(f"  Fraction: {data_fraction:.1%}")

if data_fraction < 0.5:
    print("  ⚠️  Using less than half the data - might be train split only")
else:
    print("  ✅ Using most/all of the data")

# Check date range of signals
signal_dates = []
for idx in [0, len(signals)//2, -1]:
    bar_idx = int(signals.iloc[idx]['idx'])
    if bar_idx < len(market_data):
        date = market_data.index[bar_idx]
        signal_dates.append((bar_idx, date))

print(f"\nSignal date range:")
for bar_idx, date in signal_dates:
    print(f"  Bar {bar_idx}: {date}")