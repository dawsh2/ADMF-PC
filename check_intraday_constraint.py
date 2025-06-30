#!/usr/bin/env python3
"""Check if intraday constraint is being respected."""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

results_dir = Path("config/bollinger/results/latest")

# Load position data
opens = pd.read_parquet(results_dir / "traces/portfolio/positions_open/positions_open.parquet")
closes = pd.read_parquet(results_dir / "traces/portfolio/positions_close/positions_close.parquet")

print("=== Checking Intraday Constraint ===\n")

# Parse timestamps
for df in [opens, closes]:
    df['timestamp'] = pd.to_datetime(df['ts'])
    df['time'] = df['timestamp'].dt.time
    df['date'] = df['timestamp'].dt.date

# Check for overnight positions
overnight_positions = []

for i in range(len(opens)):
    if i < len(closes):
        open_date = opens.iloc[i]['date']
        close_date = closes.iloc[i]['date']
        
        if open_date != close_date:
            overnight_positions.append({
                'trade_num': i,
                'open_date': open_date,
                'close_date': close_date,
                'open_time': opens.iloc[i]['time'],
                'close_time': closes.iloc[i]['time']
            })

print(f"Overnight positions found: {len(overnight_positions)}")

if overnight_positions:
    print("\nFirst 5 overnight positions:")
    for i, pos in enumerate(overnight_positions[:5]):
        print(f"\n{i+1}. Trade #{pos['trade_num']}:")
        print(f"   Opened: {pos['open_date']} at {pos['open_time']}")
        print(f"   Closed: {pos['close_date']} at {pos['close_time']}")

# Check for trades near end of day
print("\n=== End of Day Analysis ===")

# Group closes by time
closes['hour'] = closes['timestamp'].dt.hour
closes['minute'] = closes['timestamp'].dt.minute

# Count closes by hour
hourly_closes = closes.groupby('hour').size()
print("\nCloses by hour:")
for hour, count in hourly_closes.items():
    print(f"  {hour:02d}:00 - {hour:02d}:59: {count} closes")

# Check specifically for 15:59 closes (market close)
eod_closes = closes[(closes['hour'] == 15) & (closes['minute'] >= 55)]
print(f"\nCloses in last 5 minutes of day (15:55-15:59): {len(eod_closes)}")

# Check exit types for EOD closes
if len(eod_closes) > 0:
    print("\nEOD close exit types:")
    for i in range(min(5, len(eod_closes))):
        close = eod_closes.iloc[i]
        metadata = close.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                pass
        
        exit_type = metadata.get('exit_type', 'unknown') if isinstance(metadata, dict) else 'unknown'
        print(f"  {close['timestamp']}: {exit_type}")

# Check if we have any "eod" exit types
all_exit_types = []
for i in range(len(closes)):
    metadata = closes.iloc[i].get('metadata', {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except:
            metadata = {}
    
    if isinstance(metadata, dict):
        exit_type = metadata.get('exit_type', 'unknown')
        all_exit_types.append(exit_type)

from collections import Counter
exit_type_counts = Counter(all_exit_types)
print(f"\n=== All Exit Types ===")
for exit_type, count in exit_type_counts.items():
    print(f"  {exit_type}: {count}")