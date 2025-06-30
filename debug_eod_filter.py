#!/usr/bin/env python3
"""Debug why EOD filter is not working."""

import pandas as pd
import json

# Check metadata
with open('config/ensemble/results/latest/metadata.json') as f:
    metadata = json.load(f)
    print("=== METADATA ===")
    print(json.dumps(metadata.get('execution', {}), indent=2))
    print()

# Load signals
signals = pd.read_parquet('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
signals['ts'] = pd.to_datetime(signals['ts'])
signals['time'] = signals['ts'].dt.hour * 100 + signals['ts'].dt.minute

# Check signal pattern
print("=== SIGNAL ANALYSIS ===")
print(f"Total signals: {len(signals)}")
print(f"Date range: {signals['ts'].min()} to {signals['ts'].max()}")

# Group by hour
hourly = signals.groupby(signals['ts'].dt.hour).size()
print("\nSignals by hour:")
for hour, count in hourly.items():
    print(f"  {hour:02d}:00 - {count:4d} signals")

# Check specific times around EOD
eod_signals = signals[(signals['time'] >= 1530) & (signals['time'] <= 1600)]
print(f"\nSignals between 3:30 PM and 4:00 PM: {len(eod_signals)}")
print(eod_signals[['ts', 'val', 'time']].head(10))

# Check for force-flat signals
print("\n=== CHECKING FOR FORCE-FLAT SIGNALS ===")
# Look for patterns where signal goes to 0 at specific times
for date in signals['ts'].dt.date.unique()[:5]:  # Check first 5 days
    day_signals = signals[signals['ts'].dt.date == date]
    last_before_350 = day_signals[day_signals['time'] < 1550].iloc[-1] if len(day_signals[day_signals['time'] < 1550]) > 0 else None
    first_after_350 = day_signals[day_signals['time'] >= 1550].iloc[0] if len(day_signals[day_signals['time'] >= 1550]) > 0 else None
    
    print(f"\n{date}:")
    if last_before_350 is not None:
        print(f"  Last before 3:50 PM: {last_before_350['ts']} val={last_before_350['val']}")
    if first_after_350 is not None:
        print(f"  First at/after 3:50 PM: {first_after_350['ts']} val={first_after_350['val']}")