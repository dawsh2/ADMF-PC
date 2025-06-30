#!/usr/bin/env python3
"""
Analyze why we're seeing only 151 bars instead of 390 in the 1-minute data.
"""

import pandas as pd
from datetime import time, timedelta

# Read the 1-minute data
print("Loading SPY_1m.csv...")
df = pd.read_csv('data/SPY_1m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df = df.sort_values('timestamp')

# Extract time components
df['date'] = df['timestamp'].dt.date
df['time'] = df['timestamp'].dt.time
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

# Get a sample day
sample_day = df['date'].unique()[0]
day_df = df[df['date'] == sample_day].copy()

print(f"\nAnalyzing data for {sample_day}")
print("="*70)

# Filter to regular market hours
market_open = time(9, 30)
market_close = time(16, 0)

# Note: The data shows UTC times, so we need to check what times correspond to market hours
# If 13:30 UTC = 9:30 AM ET, then we need to adjust
market_hours_df = day_df[(day_df['hour'] >= 13) & (day_df['hour'] < 20)].copy()

print(f"\nTotal bars for the day: {len(day_df)}")
print(f"Bars during market hours (13:30-20:00 UTC): {len(market_hours_df)}")

# Check for gaps by looking at time differences
market_hours_df = market_hours_df.sort_values('timestamp')
market_hours_df['time_diff'] = market_hours_df['timestamp'].diff()

# Find gaps larger than 1 minute
gaps = market_hours_df[market_hours_df['time_diff'] > pd.Timedelta(minutes=1)]

if len(gaps) > 0:
    print(f"\n⚠️  Found {len(gaps)} gaps in the data!")
    print("\nGaps larger than 1 minute:")
    print("-"*70)
    for idx, row in gaps.iterrows():
        prev_idx = market_hours_df.index.get_loc(idx) - 1
        if prev_idx >= 0:
            prev_row = market_hours_df.iloc[prev_idx]
            gap_minutes = row['time_diff'].total_seconds() / 60
            print(f"Gap: {prev_row['timestamp']} → {row['timestamp']} ({gap_minutes:.0f} minutes)")

# Count expected vs actual bars
print("\n" + "="*70)
print("EXPECTED VS ACTUAL BARS")
print("="*70)

# Group by hour to see distribution
hourly_counts = market_hours_df.groupby('hour').size()
print("\nBars per hour (UTC):")
for hour, count in hourly_counts.items():
    expected = 60  # Should have 60 bars per hour for 1-minute data
    print(f"  {hour:02d}:00 - {hour:02d}:59 → {count} bars (expected: {expected})")

# Check specific time ranges
print("\n" + "="*70)
print("DETAILED MINUTE-BY-MINUTE CHECK")
print("="*70)

# Look at the first hour of trading (9:30-10:30 AM ET = 13:30-14:30 UTC)
first_hour = market_hours_df[(market_hours_df['hour'] == 13) | 
                             ((market_hours_df['hour'] == 14) & (market_hours_df['minute'] < 30))]

print(f"\nFirst hour of trading (9:30-10:30 AM ET):")
print(f"Total bars: {len(first_hour)} (expected: 60)")

# Check which minutes are present
minutes_present = set()
for _, row in first_hour.iterrows():
    minutes_present.add((row['hour'], row['minute']))

# Find missing minutes
print("\nMissing minutes in first hour:")
missing_count = 0
for h in [13, 14]:
    for m in range(60):
        if h == 14 and m >= 30:
            break
        if h == 13 and m < 30:
            continue
        if (h, m) not in minutes_present:
            print(f"  Missing: {h:02d}:{m:02d}")
            missing_count += 1

if missing_count == 0:
    print("  None - all minutes present!")

# Sample the actual timestamps
print("\n" + "="*70)
print("SAMPLE TIMESTAMPS FROM DATA")
print("="*70)

print("\nFirst 10 bars of the day:")
for i, (_, row) in enumerate(market_hours_df.head(10).iterrows()):
    print(f"  Bar {i}: {row['timestamp']}")

print("\nLast 10 bars before 4:00 PM:")
before_close = market_hours_df[market_hours_df['hour'] < 20]
for i, (_, row) in enumerate(before_close.tail(10).iterrows()):
    print(f"  Bar {i + len(before_close) - 10}: {row['timestamp']}")

# Final summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

actual_bars = len(market_hours_df[market_hours_df['hour'] < 20])
expected_bars = 390  # 6.5 hours * 60 minutes

print(f"\nFor {sample_day}:")
print(f"  Expected bars (9:30 AM - 4:00 PM): {expected_bars}")
print(f"  Actual bars found: {actual_bars}")
print(f"  Missing bars: {expected_bars - actual_bars}")
print(f"  Data completeness: {actual_bars/expected_bars*100:.1f}%")

# Check if this pattern is consistent across days
print("\n" + "="*70)
print("CHECKING OTHER DAYS")
print("="*70)

for day in df['date'].unique()[:10]:
    day_df = df[df['date'] == day]
    market_df = day_df[(day_df['hour'] >= 13) & (day_df['hour'] < 20)]
    regular_hours = market_df[market_df['time'] <= time(20, 0)]
    print(f"{day}: {len(regular_hours)} bars")