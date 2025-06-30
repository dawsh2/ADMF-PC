#!/usr/bin/env python3
"""Verify what timeframe the data actually is."""

import pandas as pd

# Load the actual data files
spy_5m = pd.read_csv('data/SPY_5m.csv', parse_dates=['timestamp'])
spy_1m = pd.read_csv('data/SPY_1m.csv', parse_dates=['timestamp'])

print("Data Timeframe Verification")
print("=" * 50)

print("\nSPY_5m.csv:")
print(f"  Total bars: {len(spy_5m):,}")
print(f"  First 5 timestamps:")
for i in range(min(5, len(spy_5m))):
    print(f"    {spy_5m.iloc[i]['timestamp']}")

# Calculate time between bars
if len(spy_5m) > 1:
    time_diff = (spy_5m.iloc[1]['timestamp'] - spy_5m.iloc[0]['timestamp']).total_seconds() / 60
    print(f"  Minutes between bars: {time_diff}")

print("\nSPY_1m.csv:")
print(f"  Total bars: {len(spy_1m):,}")
print(f"  First 5 timestamps:")
for i in range(min(5, len(spy_1m))):
    print(f"    {spy_1m.iloc[i]['timestamp']}")

# Calculate time between bars
if len(spy_1m) > 1:
    time_diff = (spy_1m.iloc[1]['timestamp'] - spy_1m.iloc[0]['timestamp']).total_seconds() / 60
    print(f"  Minutes between bars: {time_diff}")

# Check which file has ~16,610 bars (80% would be ~20,762 total)
print(f"\n80% of SPY_5m: {int(len(spy_5m) * 0.8)} bars")
print(f"80% of SPY_1m: {int(len(spy_1m) * 0.8)} bars")

print("\nConclusion:")
if abs(int(len(spy_5m) * 0.8) - 16610) < 100:
    print("✅ You ARE using 5-minute data (SPY_5m.csv)")
    print("   The source file reference is just incorrectly labeled")
elif abs(int(len(spy_1m) * 0.8) - 16610) < 100:
    print("❌ You ARE using 1-minute data (SPY_1m.csv)")
    print("   This would explain the performance difference!")
else:
    print("❓ Can't determine which file is being used")