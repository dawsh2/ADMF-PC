#!/usr/bin/env python3
"""Check if EOD (end-of-day) exit logic is working."""

import pandas as pd
import json

# Load signals
signals = pd.read_parquet('config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')

# Convert timestamps
signals['ts'] = pd.to_datetime(signals['ts'])

# Extract time and check for EOD exits
signals['time'] = signals['ts'].dt.time
signals['date'] = signals['ts'].dt.date

print('=== EOD EXIT VERIFICATION ===')
print(f'Total signal changes: {len(signals)}')

# Check for positions held past 3:55 PM (15:55)
eod_threshold = pd.to_datetime('15:55:00').time()
late_signals = signals[signals['time'] >= eod_threshold]

print(f'\nSignals at/after 3:55 PM: {len(late_signals)}')
if len(late_signals) > 0:
    print('\nLate signal examples:')
    for _, row in late_signals.head(10).iterrows():
        print(f'  {row["ts"]}: signal = {row["val"]}')

# Check for overnight positions
print('\n=== OVERNIGHT POSITION CHECK ===')
# Group by date and check last signal of each day
daily_last_signal = signals.groupby('date').agg({
    'ts': 'last',
    'val': 'last',
    'time': 'last'
}).reset_index()

# Count days ending with open positions
overnight_positions = daily_last_signal[daily_last_signal['val'] != 0]
print(f'Days ending with open positions: {len(overnight_positions)} out of {len(daily_last_signal)} days')

if len(overnight_positions) > 0:
    print('\nDays with overnight positions (first 10):')
    for _, row in overnight_positions.head(10).iterrows():
        print(f'  {row["date"]}: Last signal at {row["time"]} = {row["val"]}')

# Check if positions are closed by EOD
print('\n=== EOD CLOSURE PATTERN CHECK ===')
# Look for pattern: position != 0 followed by 0 at end of day
eod_closures = 0
for date in signals['date'].unique()[:20]:  # Check first 20 days
    day_signals = signals[signals['date'] == date].sort_values('ts')
    if len(day_signals) >= 2:
        # Check if last signal is a close (0) and previous was open
        last_signal = day_signals.iloc[-1]
        if last_signal['val'] == 0 and len(day_signals) > 1:
            prev_signal = day_signals.iloc[-2]
            if prev_signal['val'] != 0 and last_signal['time'] >= pd.to_datetime('15:55:00').time():
                eod_closures += 1
                
print(f'Days with EOD closure pattern: {eod_closures}')

# Check command used
print('\n=== CHECKING RUN COMMAND ===')
try:
    # Try to find if --close-eod was used
    with open('config/ensemble/results/latest/metadata.json') as f:
        metadata = json.load(f)
    # The metadata doesn't store the command, but we can infer from behavior
    if len(overnight_positions) == 0:
        print('✅ No overnight positions found - EOD exits likely working')
    else:
        print('⚠️  Overnight positions found - EOD exits may not be enabled')
except:
    pass