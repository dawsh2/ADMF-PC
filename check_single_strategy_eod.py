#!/usr/bin/env python3
"""Check EOD for single strategy output."""

import pandas as pd
import os

# Find the parquet file
parquet_dir = 'config/results/latest/traces'
parquet_file = None
for root, dirs, files in os.walk(parquet_dir):
    for f in files:
        if f.endswith('.parquet'):
            parquet_file = os.path.join(root, f)
            break
    if parquet_file:
        break

if not parquet_file:
    print("No parquet file found!")
    exit(1)

print(f"Loading signals from: {parquet_file}")

# Load signals
signals = pd.read_parquet(parquet_file)
print(f"Total signal changes: {len(signals)}")

# Convert timestamps
signals['ts'] = pd.to_datetime(signals['ts'])
signals['time'] = signals['ts'].dt.time
signals['date'] = signals['ts'].dt.date
signals['hour'] = signals['ts'].dt.hour
signals['minute'] = signals['ts'].dt.minute

# Calculate bar_of_day 
signals['minutes_since_930'] = (signals['hour'] - 9) * 60 + signals['minute'] - 30
signals['bar_of_day'] = signals['minutes_since_930'] / 5

print(f"\nBar of day range: {signals['bar_of_day'].min():.0f} to {signals['bar_of_day'].max():.0f}")

# Check for signals after bar 78
late_signals = signals[signals['bar_of_day'] >= 78]
print(f"\nSignals at/after bar 78 (3:50 PM): {len(late_signals)}")

if len(late_signals) > 0:
    non_zero_late = late_signals[late_signals['val'] != 0]
    print(f"Non-zero signals after bar 78: {len(non_zero_late)}")
    
    if len(non_zero_late) == 0:
        print("✅ SUCCESS: All signals after 3:50 PM are flat (0)")
    else:
        print("❌ FAILURE: Found non-zero signals after 3:50 PM")
        print(non_zero_late[['ts', 'val', 'bar_of_day']].head())

# Check overnight positions
daily_last = signals.groupby('date')['val'].last()
overnight = daily_last[daily_last != 0]
print(f"\nDays ending with open positions: {len(overnight)} out of {len(daily_last)}")

if len(overnight) == 0:
    print("✅ SUCCESS: No overnight positions!")
else:
    print("⚠️  Still have overnight positions")