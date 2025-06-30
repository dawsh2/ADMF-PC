#!/usr/bin/env python3
"""
Check why bar_of_day calculation showed only 151 bars earlier.
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
sample_day = df['date'].unique()[0]
print(f"Analyzing {sample_day}")
print("="*70)

# Get all data for the day
full_day_df = df[df['date'] == sample_day].copy()
print(f"\nAll bars for {sample_day}: {len(full_day_df)}")

# Check time range
print(f"Time range: {full_day_df['time'].min()} to {full_day_df['time'].max()}")

# Now let's properly filter for market hours
# The data is in UTC, where 13:30 = 9:30 AM ET and 20:00 = 4:00 PM ET
market_open_utc = time(13, 30)  # 9:30 AM ET
market_close_utc = time(20, 0)   # 4:00 PM ET

# Filter to market hours
market_hours_df = full_day_df[(full_day_df['time'] >= market_open_utc) & 
                              (full_day_df['time'] <= market_close_utc)].copy()

print(f"\nMarket hours bars (13:30-20:00 UTC = 9:30 AM-4:00 PM ET): {len(market_hours_df)}")

# Calculate bar_of_day for market hours
market_hours_df = market_hours_df.sort_values('timestamp')
market_hours_df['bar_of_day'] = range(len(market_hours_df))

# Find key times
key_times = [
    (time(15, 30), "3:30 PM ET (15:30 UTC)"),
    (time(15, 50), "3:50 PM ET (15:50 UTC)"),  
    (time(16, 0), "4:00 PM ET (16:00 UTC)"),
    (time(19, 30), "3:30 PM ET (19:30 UTC)"),
    (time(19, 50), "3:50 PM ET (19:50 UTC)"),
    (time(20, 0), "4:00 PM ET (20:00 UTC)")
]

print("\n" + "="*70)
print("KEY TIME ANALYSIS")
print("="*70)

for check_time, description in key_times:
    matches = market_hours_df[market_hours_df['time'] == check_time]
    if len(matches) > 0:
        bar_num = matches.iloc[0]['bar_of_day']
        print(f"\n{description}:")
        print(f"  Timestamp: {matches.iloc[0]['timestamp']}")
        print(f"  Bar of day: {bar_num}")
    else:
        print(f"\n{description}: NOT FOUND")

# The confusion might be UTC vs ET
print("\n" + "="*70)
print("UTC TO ET CONVERSION")
print("="*70)
print("\nThe data uses UTC timestamps!")
print("- 13:30 UTC = 9:30 AM ET (market open)")
print("- 19:30 UTC = 3:30 PM ET (entry cutoff)")
print("- 19:50 UTC = 3:50 PM ET (force exit)")
print("- 20:00 UTC = 4:00 PM ET (market close)")

# Recalculate with correct UTC times
entry_cutoff_utc = time(19, 30)  # 3:30 PM ET
force_exit_utc = time(19, 50)    # 3:50 PM ET

entry_bar = market_hours_df[market_hours_df['time'] == entry_cutoff_utc]['bar_of_day'].values
exit_bar = market_hours_df[market_hours_df['time'] == force_exit_utc]['bar_of_day'].values

print("\n" + "="*70)
print("CORRECTED EOD BAR CALCULATIONS")
print("="*70)

if len(entry_bar) > 0:
    print(f"\n✓ Entry cutoff (3:30 PM ET = 19:30 UTC) is bar_of_day {entry_bar[0]}")
    print(f"  Expected: 360 (6 hours * 60 minutes after market open)")
    
if len(exit_bar) > 0:
    print(f"\n✓ Force exit (3:50 PM ET = 19:50 UTC) is bar_of_day {exit_bar[0]}")
    print(f"  Expected: 380 (6 hours 20 minutes * 60 after market open)")

# Show some bars around EOD
print("\n" + "="*70)
print("BARS AROUND EOD (3:30-4:00 PM ET)")
print("="*70)

eod_bars = market_hours_df[(market_hours_df['time'] >= time(19, 25)) & 
                           (market_hours_df['time'] <= time(20, 0))]

print("\nTime (UTC) | Bar # | Price")
print("-"*40)
for _, row in eod_bars.head(15).iterrows():
    print(f"{row['time']} | {row['bar_of_day']:>3} | ${row['Close']:.2f}")

# Final check on bar_of_day calculation
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)
print(f"\nTotal market hours bars: {len(market_hours_df)}")
print(f"Should be: 390 (6.5 hours * 60 minutes)")
print(f"Match: {'✓' if len(market_hours_df) == 390 else '✗'}")