#!/usr/bin/env python3
"""
Check alignment between signal and market data files
"""

import pandas as pd

# Read both files
signal_file = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_47e9dad1/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_35_11.parquet"
market_file = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"

print("Reading signal file...")
signals_df = pd.read_parquet(signal_file)
print(f"Signal file shape: {signals_df.shape}")
print(f"Signal file columns: {list(signals_df.columns)}")
print(f"First signal timestamp: {signals_df['ts'].min()}")
print(f"First signal idx: {signals_df['idx'].min()}")

print("\nReading market data...")
market_df = pd.read_parquet(market_file)
print(f"Market data shape: {market_df.shape}")
print(f"Market data columns: {list(market_df.columns)}")
print(f"First market timestamp: {market_df['timestamp'].min()}")
print(f"First market bar_index: {market_df['bar_index'].min()}")

# Check the timestamp at index 29/30
print("\n=== Checking alignment at index 29/30 ===")
print(f"\nMarket data at bar_index 29:")
print(market_df[market_df['bar_index'] == 29][['bar_index', 'timestamp', 'close']])

print(f"\nMarket data at bar_index 30:")
print(market_df[market_df['bar_index'] == 30][['bar_index', 'timestamp', 'close']])

print(f"\nSignal data at idx 30:")
print(signals_df[signals_df['idx'] == 30][['idx', 'ts', 'val', 'px']])

# Check if px field in signal matches any price in market data
if 'px' in signals_df.columns:
    print("\n=== Checking px field ===")
    first_signal = signals_df.iloc[0]
    print(f"First signal px: {first_signal['px']}")
    print(f"Signal timestamp: {first_signal['ts']}")
    
    # Find market data at same timestamp
    signal_ts = pd.to_datetime(first_signal['ts'])
    market_at_signal = market_df[market_df['timestamp'] == signal_ts]
    if not market_at_signal.empty:
        print(f"\nMarket data at signal timestamp:")
        print(market_at_signal[['timestamp', 'open', 'high', 'low', 'close']])
    
# Check time differences
print("\n=== Time Analysis ===")
print(f"Market data timezone: {market_df['timestamp'].iloc[0].tzinfo}")
print(f"Signal data timezone: {pd.to_datetime(signals_df['ts'].iloc[0]).tzinfo}")

# Convert to same timezone for comparison
market_df['timestamp_utc'] = pd.to_datetime(market_df['timestamp']).dt.tz_convert('UTC')
signals_df['ts_utc'] = pd.to_datetime(signals_df['ts']).dt.tz_convert('UTC')

# Find first few matching timestamps
print("\n=== First 5 signals with matching market data ===")
for i in range(min(5, len(signals_df))):
    signal = signals_df.iloc[i]
    signal_ts = signal['ts_utc']
    market_match = market_df[market_df['timestamp_utc'] == signal_ts]
    
    if not market_match.empty:
        print(f"\nSignal idx={signal['idx']}, ts={signal['ts']}, val={signal['val']}")
        print(f"Market bar_index={market_match.iloc[0]['bar_index']}, close={market_match.iloc[0]['close']}")
        print(f"Index difference: signal_idx - market_bar_index = {signal['idx'] - market_match.iloc[0]['bar_index']}")