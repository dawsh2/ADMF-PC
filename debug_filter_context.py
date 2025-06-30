#!/usr/bin/env python3
"""Debug what's happening with the filter context."""

import pandas as pd

# Load the actual data to see what timestamps we have
data = pd.read_csv('data/SPY_5m.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.hour
data['minute'] = data['timestamp'].dt.minute
data['time'] = data['hour'] * 100 + data['minute']

print("=== DATA ANALYSIS ===")
print(f"Total bars: {len(data)}")
print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
print(f"\nTime range per day:")
print(f"Earliest time: {data['time'].min()} ({data['time'].min()//100}:{data['time'].min()%100:02d})")
print(f"Latest time: {data['time'].max()} ({data['time'].max()//100}:{data['time'].max()%100:02d})")

# Check how many bars are after 3:50 PM
late_bars = data[data['time'] >= 1550]
print(f"\nBars at/after 3:50 PM: {len(late_bars)} ({len(late_bars)/len(data)*100:.1f}%)")

# Sample some late bars
print("\nSample late bars:")
print(late_bars[['timestamp', 'time', 'close']].head(10))

# Check a specific day
day_data = data[data['timestamp'].dt.date == pd.to_datetime('2024-03-26').date()]
print(f"\n=== 2024-03-26 Analysis ===")
print(f"Bars for this day: {len(day_data)}")
print(f"Time range: {day_data['time'].min()} to {day_data['time'].max()}")
print("\nBars after 3:50 PM:")
print(day_data[day_data['time'] >= 1550][['timestamp', 'time', 'close']])