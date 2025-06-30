#!/usr/bin/env python3
"""
Check the actual session times in the 1-minute data.
"""

import pandas as pd
from datetime import time

# Read the 1-minute data
df = pd.read_csv('data/SPY_1m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df = df.sort_values('timestamp')

# Extract time components
df['date'] = df['timestamp'].dt.date
df['time'] = df['timestamp'].dt.time
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

# Get a sample day
sample_days = df['date'].unique()[:5]

print("Session Analysis for 1-minute SPY data")
print("="*70)

for day in sample_days:
    day_df = df[df['date'] == day].copy()
    day_df = day_df.sort_values('timestamp')
    
    # Find first and last bar of the day
    first_bar = day_df.iloc[0]
    last_bar = day_df.iloc[-1]
    
    print(f"\n{day}:")
    print(f"  First bar: {first_bar['time']} ({first_bar['timestamp']})")
    print(f"  Last bar:  {last_bar['time']} ({last_bar['timestamp']})")
    print(f"  Total bars: {len(day_df)}")
    
    # Calculate minutes from first bar to key times
    if len(day_df[day_df['time'] == time(15, 30)]) > 0:
        # Find the bar number for 3:30 PM
        day_df['bar_num'] = range(len(day_df))
        bar_330 = day_df[day_df['time'] == time(15, 30)]['bar_num'].values[0]
        bar_350 = day_df[day_df['time'] == time(15, 50)]['bar_num'].values[0]
        bar_400 = day_df[day_df['time'] == time(16, 0)]['bar_num'].values[0] if len(day_df[day_df['time'] == time(16, 0)]) > 0 else 'N/A'
        
        print(f"  3:30 PM is bar #{bar_330}")
        print(f"  3:50 PM is bar #{bar_350}")
        print(f"  4:00 PM is bar #{bar_400}")

print("\n" + "="*70)
print("KEY FINDING:")
print("="*70)
print("\nThe 1-minute data appears to start BEFORE regular market hours!")
print("This explains why 3:30 PM is bar 120 instead of bar 360.")
print("\nFor EOD closing with 1-minute data that includes pre-market:")
print("- We need to calculate bar_of_day from the actual first bar")
print("- Or filter to only regular market hours (9:30 AM - 4:00 PM)")

# Check the actual time range
all_times = sorted(df['time'].unique())
print(f"\nTime range in data: {all_times[0]} to {all_times[-1]}")

# Count pre-market and after-hours bars
pre_market = df[df['time'] < time(9, 30)]
after_hours = df[df['time'] > time(16, 0)]
regular_hours = df[(df['time'] >= time(9, 30)) & (df['time'] <= time(16, 0))]

print(f"\nBar distribution:")
print(f"  Pre-market bars (before 9:30 AM): {len(pre_market):,}")
print(f"  Regular hours bars (9:30 AM - 4:00 PM): {len(regular_hours):,}")
print(f"  After-hours bars (after 4:00 PM): {len(after_hours):,}")

# Calculate correct bar numbers for regular market hours only
print("\n" + "="*70)
print("CORRECTED BAR NUMBERS (Regular Market Hours Only)")
print("="*70)

sample_day = sample_days[0]
day_df = df[df['date'] == sample_day].copy()
# Filter to regular market hours
day_df = day_df[(day_df['time'] >= time(9, 30)) & (day_df['time'] <= time(16, 0))]
day_df = day_df.sort_values('timestamp')
day_df['bar_of_day'] = range(len(day_df))

if len(day_df[day_df['time'] == time(15, 30)]) > 0:
    bar_330 = day_df[day_df['time'] == time(15, 30)]['bar_of_day'].values[0]
    bar_350 = day_df[day_df['time'] == time(15, 50)]['bar_of_day'].values[0]
    bar_400 = day_df[day_df['time'] == time(16, 0)]['bar_of_day'].values[0]
    
    print(f"\nWith regular market hours only (9:30 AM - 4:00 PM):")
    print(f"  3:30 PM is bar_of_day {bar_330} (expected: 360)")
    print(f"  3:50 PM is bar_of_day {bar_350} (expected: 380)")
    print(f"  4:00 PM is bar_of_day {bar_400} (expected: 390)")
    
    # Check if our expectations match
    if bar_330 == 360:
        print("\n✓ Bar numbers match expected values!")
    else:
        print(f"\n⚠️ Bar numbers don't match. Difference might be due to missing bars.")