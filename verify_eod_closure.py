#!/usr/bin/env python3
"""Verify EOD closure with time-based filter."""

import pandas as pd
import glob

# Find most recent result directory
trace_files = glob.glob('config/ensemble/results/*/traces/*/SPY_5m_compiled_strategy_0.parquet')
if not trace_files:
    print("No trace files found yet")
    exit(1)
    
# Get the most recent one
trace_files.sort()
latest_file = trace_files[-1]
print(f"Checking file: {latest_file}")

# Load signals
signals = pd.read_parquet(latest_file)

# Convert timestamps
signals['ts'] = pd.to_datetime(signals['ts'])
signals['time'] = signals['ts'].dt.time
signals['date'] = signals['ts'].dt.date
signals['hour'] = signals['ts'].dt.hour
signals['minute'] = signals['ts'].dt.minute
signals['time_hhmm'] = signals['hour'] * 100 + signals['minute']

print('\n=== TIME-BASED EOD VERIFICATION ===')
print(f'Total signal changes: {len(signals)}')
print(f'Date range: {signals["date"].min()} to {signals["date"].max()}')

# Check for signals at/after 3:50 PM (1550)
late_signals = signals[signals['time_hhmm'] >= 1550]
print(f'\nSignals at/after 3:50 PM (time >= 1550): {len(late_signals)}')

if len(late_signals) > 0:
    non_zero_late = late_signals[late_signals['val'] != 0]
    print(f'Non-zero signals after 3:50 PM: {len(non_zero_late)}')
    
    if len(non_zero_late) == 0:
        print('✅ SUCCESS: All signals after 3:50 PM are flat (0)')
    else:
        print('❌ FAILURE: Found non-zero signals after 3:50 PM:')
        print(non_zero_late[['ts', 'val', 'time_hhmm']].head(10))
else:
    print('✅ SUCCESS: No signals generated after 3:50 PM!')

# Check overnight positions
print('\n=== OVERNIGHT POSITION CHECK ===')
daily_last_signal = signals.groupby('date').agg({
    'ts': 'last',
    'val': 'last',
    'time': 'last',
    'time_hhmm': 'last'
}).reset_index()

overnight_positions = daily_last_signal[daily_last_signal['val'] != 0]
print(f'Days ending with open positions: {len(overnight_positions)} out of {len(daily_last_signal)} days')

if len(overnight_positions) == 0:
    print('✅ SUCCESS: No overnight positions!')
else:
    print('\nDays with overnight positions (sample):')
    for _, row in overnight_positions.head(10).iterrows():
        print(f'  {row["date"]}: Last signal at {row["time"]} (time={row["time_hhmm"]}) = {row["val"]}')
    
    # Check if any are after our cutoff
    late_overnight = overnight_positions[overnight_positions['time_hhmm'] >= 1550]
    if len(late_overnight) > 0:
        print(f'\n⚠️  {len(late_overnight)} days have positions after 3:50 PM cutoff')

# Summary
print('\n=== SUMMARY ===')
non_zero_late = late_signals[late_signals['val'] != 0] if len(late_signals) > 0 else pd.DataFrame()
if len(non_zero_late) == 0 and len(overnight_positions) == 0:
    print('✅ EOD closure is working perfectly!')
elif len(non_zero_late) == 0:
    print('✅ Time filter is working (no signals after 3:50 PM)')
    print('⚠️  But still have overnight positions from earlier signals')
else:
    print('❌ Time filter is NOT working correctly')