#!/usr/bin/env python3
"""Check if P=20, M=3.0 strategy holds positions overnight"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load trace
trace_file = Path("config/keltner/robust_config/results/20250622_213055/traces/mean_reversion/SPY_5m_kb_robust_p10_m3.parquet")
df_trace = pd.read_parquet(trace_file)

# Load SPY data to get timestamps
spy_data = pd.read_csv("data/SPY_5m.csv")
spy_data['datetime'] = pd.to_datetime(spy_data['timestamp'])
spy_data['time'] = spy_data['datetime'].dt.time
spy_data['date'] = spy_data['datetime'].dt.date

print("OVERNIGHT HOLDING ANALYSIS")
print("="*60)

# Analyze each trade
overnight_trades = 0
intraday_trades = 0
long_durations = []

for i in range(len(df_trace) - 1):
    if df_trace.iloc[i]['val'] != 0:  # Entry
        entry_idx = df_trace.iloc[i]['idx']
        entry_date = spy_data.iloc[entry_idx]['date']
        entry_time = spy_data.iloc[entry_idx]['time']
        
        # Find exit
        j = i + 1
        while j < len(df_trace) and df_trace.iloc[j]['val'] != 0:
            j += 1
            
        if j < len(df_trace):
            exit_idx = df_trace.iloc[j]['idx']
            exit_date = spy_data.iloc[exit_idx]['date']
            exit_time = spy_data.iloc[exit_idx]['time']
            
            # Check if overnight
            if entry_date != exit_date:
                overnight_trades += 1
                print(f"OVERNIGHT: Entry {entry_date} {entry_time} -> Exit {exit_date} {exit_time}")
            else:
                intraday_trades += 1
                
            # Duration in bars
            duration = exit_idx - entry_idx
            if duration > 78:  # More than one day
                long_durations.append((duration, entry_date, exit_date))

print(f"\nTotal trades analyzed: {overnight_trades + intraday_trades}")
print(f"Overnight holds: {overnight_trades} ({overnight_trades/(overnight_trades + intraday_trades)*100:.1f}%)")
print(f"Intraday only: {intraday_trades}")

if long_durations:
    print(f"\nTrades longer than 1 day: {len(long_durations)}")
    for duration, entry, exit in long_durations[:5]:
        print(f"  {duration} bars ({duration/78:.1f} days): {entry} -> {exit}")