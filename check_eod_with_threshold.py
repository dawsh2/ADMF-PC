#!/usr/bin/env python3
"""Check if EOD threshold is working."""

import pandas as pd
import json

# Load signals - try both possible paths
try:
    signals = pd.read_parquet('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
except FileNotFoundError:
    signals = pd.read_parquet('config/ensemble/results/latest/traces/composite/SPY_5m_compiled_strategy_0.parquet')

# Convert timestamps
signals['ts'] = pd.to_datetime(signals['ts'])
signals['time'] = signals['ts'].dt.time
signals['date'] = signals['ts'].dt.date

# Add bar_of_day calculation (assuming 5-minute bars starting at 9:30 AM EST)
signals['hour'] = signals['ts'].dt.hour
signals['minute'] = signals['ts'].dt.minute
signals['minutes_since_930'] = (signals['hour'] - 9) * 60 + signals['minute'] - 30
signals['bar_of_day'] = signals['minutes_since_930'] / 5

print('=== EOD THRESHOLD VERIFICATION ===')
print(f'Total signal changes: {len(signals)}')

# Check for signals after bar 78 (3:50 PM)
late_signals = signals[signals['bar_of_day'] >= 78]
print(f'\nSignals at/after bar 78 (3:50 PM): {len(late_signals)}')

if len(late_signals) > 0:
    print('\nLate signals (should all be 0):')
    print(late_signals[['ts', 'val', 'bar_of_day']].head(10))
    non_zero_late = late_signals[late_signals['val'] != 0]
    if len(non_zero_late) > 0:
        print(f'\n❌ ERROR: Found {len(non_zero_late)} non-zero signals after bar 78!')
        print(non_zero_late[['ts', 'val', 'bar_of_day']].head())
    else:
        print('\n✅ All signals after bar 78 are 0 (positions closed)')

# Check overnight positions
print('\n=== OVERNIGHT POSITION CHECK ===')
daily_last_signal = signals.groupby('date').agg({
    'ts': 'last',
    'val': 'last',
    'time': 'last',
    'bar_of_day': 'last'
}).reset_index()

overnight_positions = daily_last_signal[daily_last_signal['val'] != 0]
print(f'Days ending with open positions: {len(overnight_positions)} out of {len(daily_last_signal)} days')

if len(overnight_positions) > 0:
    print('\nDays with overnight positions (first 10):')
    for _, row in overnight_positions.head(10).iterrows():
        print(f'  {row["date"]}: Last signal at {row["time"]} (bar {row["bar_of_day"]:.0f}) = {row["val"]}')