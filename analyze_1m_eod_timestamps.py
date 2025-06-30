#!/usr/bin/env python3
"""
Analyze EOD timestamps in the 1-minute SPY data file.
"""

import pandas as pd
from datetime import time

# Read the 1-minute data
print("Loading SPY_1m.csv...")
df = pd.read_csv('data/SPY_1m.csv')

# Parse timestamps
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df = df.sort_values('timestamp')

# Extract time components
df['date'] = df['timestamp'].dt.date
df['time'] = df['timestamp'].dt.time
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

print(f"\nTotal bars: {len(df):,}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Get unique trading days
unique_dates = df['date'].unique()
print(f"\nUnique trading days: {len(unique_dates)}")

# Analyze EOD patterns
print("\n" + "="*80)
print("EOD TIMESTAMP ANALYSIS (3:00 PM - 4:00 PM)")
print("="*80)

# Look at bars between 3:00 PM and 4:00 PM
eod_df = df[(df['hour'] == 15) | ((df['hour'] == 16) & (df['minute'] == 0))]

# Group by time to see frequency
time_counts = eod_df.groupby('time').size().sort_index()
print("\nBars per timestamp (3:00 PM - 4:00 PM):")
print("-"*40)
for t, count in time_counts.items():
    print(f"{t} → {count} bars")

# Check specific critical times
critical_times = [
    (time(15, 30), "3:30 PM - Entry cutoff"),
    (time(15, 50), "3:50 PM - Force exit"),
    (time(15, 55), "3:55 PM"),
    (time(16, 0), "4:00 PM - Market close")
]

print("\n" + "="*80)
print("CRITICAL TIME COVERAGE")
print("="*80)

for check_time, description in critical_times:
    count = len(df[df['time'] == check_time])
    days_with_time = len(df[df['time'] == check_time]['date'].unique())
    print(f"\n{description} ({check_time}):")
    print(f"  - Total bars: {count}")
    print(f"  - Days with this timestamp: {days_with_time}")
    
    # Show sample timestamps
    sample = df[df['time'] == check_time].head(5)
    if len(sample) > 0:
        print("  - Sample timestamps:")
        for _, row in sample.iterrows():
            print(f"    {row['timestamp']}")

# Check for gaps in the last 30 minutes of trading
print("\n" + "="*80)
print("GAP ANALYSIS (3:30 PM - 4:00 PM)")
print("="*80)

# For each day, check if we have continuous bars from 3:30 to 4:00
sample_days = unique_dates[:5]  # Check first 5 days
for day in sample_days:
    day_df = df[df['date'] == day]
    eod_bars = day_df[(day_df['hour'] == 15) & (day_df['minute'] >= 30)]
    
    print(f"\n{day}:")
    print(f"  - Bars from 3:30 PM onwards: {len(eod_bars)}")
    
    # Check for specific times
    has_330 = any(day_df['time'] == time(15, 30))
    has_350 = any(day_df['time'] == time(15, 50))
    has_400 = any(day_df['time'] == time(16, 0))
    
    print(f"  - Has 3:30 PM: {'✓' if has_330 else '✗'}")
    print(f"  - Has 3:50 PM: {'✓' if has_350 else '✗'}")
    print(f"  - Has 4:00 PM: {'✓' if has_400 else '✗'}")

# Calculate bar_of_day for verification
print("\n" + "="*80)
print("BAR_OF_DAY CALCULATION VERIFICATION")
print("="*80)

# Take a sample day and calculate bar_of_day
sample_day = unique_dates[0]
day_df = df[df['date'] == sample_day].copy()

# Filter to market hours only (9:30 AM to 4:00 PM)
market_open = time(9, 30)
market_close = time(16, 0)
day_df = day_df[(day_df['time'] >= market_open) & (day_df['time'] <= market_close)]

# Calculate bar_of_day (0-based)
day_df = day_df.sort_values('timestamp')
day_df['bar_of_day'] = range(len(day_df))

print(f"\nSample day: {sample_day}")
print(f"Market hours bars: {len(day_df)}")

# Show bars around critical times
critical_bars = day_df[(day_df['time'] >= time(15, 25)) & (day_df['time'] <= time(16, 0))]
print("\nBars around EOD:")
print("-"*60)
print("Time     | Bar # | Bar_of_day | Price")
print("-"*60)
for _, row in critical_bars.iterrows():
    print(f"{row['time']} | {row.name:<5} | {row['bar_of_day']:<10} | ${row['Close']:.2f}")

# Verify our calculations
bar_330 = day_df[day_df['time'] == time(15, 30)]['bar_of_day'].values
bar_350 = day_df[day_df['time'] == time(15, 50)]['bar_of_day'].values

if len(bar_330) > 0:
    print(f"\n✓ 3:30 PM is bar_of_day {bar_330[0]} (we expect 360)")
if len(bar_350) > 0:
    print(f"✓ 3:50 PM is bar_of_day {bar_350[0]} (we expect 380)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nFor 1-minute data:")
print("- Entry cutoff (3:30 PM) should be bar_of_day < 360")
print("- Force exit (3:50 PM) should be bar_of_day >= 380")
print("\nNote: 3:50 PM is 380 minutes after 9:30 AM, not 390.")
print("390 would be 4:00 PM exactly.")