#!/usr/bin/env python3
"""
Check timezone of timestamps in parquet files
"""

import pandas as pd
from pathlib import Path

# Load a sample parquet file
parquet_file = Path('config/bollinger/results/20250624_150142/traces/SPY_5m_1m/signals/bollinger_bands/SPY_5m_compiled_strategy_911.parquet')

if parquet_file.exists():
    # Load signals
    signals = pd.read_parquet(parquet_file)
    signals['ts'] = pd.to_datetime(signals['ts'])
    
    print("🕐 Timezone Analysis")
    print("="*60)
    
    # Check timezone info
    print(f"Timezone aware: {signals['ts'].dt.tz is not None}")
    if signals['ts'].dt.tz is not None:
        print(f"Timezone: {signals['ts'].dt.tz}")
    
    # Show first and last few timestamps
    print("\nFirst 5 timestamps:")
    for ts in signals['ts'].head():
        print(f"  {ts}")
    
    print("\nLast 5 timestamps:")
    for ts in signals['ts'].tail():
        print(f"  {ts}")
    
    # Group by date and show the time range for a few days
    signals['date'] = signals['ts'].dt.date
    signals['time'] = signals['ts'].dt.time
    
    print("\n📅 Daily trading time ranges (first 10 days):")
    for i, (date, day_signals) in enumerate(signals.groupby('date')):
        if i >= 10:
            break
        min_time = day_signals['time'].min()
        max_time = day_signals['time'].max()
        print(f"  {date}: {min_time} - {max_time}")
    
    # Check what hours we see trades
    signals['hour'] = signals['ts'].dt.hour
    hour_counts = signals['hour'].value_counts().sort_index()
    
    print("\n📊 Signal distribution by hour:")
    for hour, count in hour_counts.items():
        print(f"  {hour:02d}:00 - {count} signals")
    
    # Look for specific times around market open/close
    print("\n🔍 Checking key market times:")
    # If UTC, 9:30 AM ET = 13:30 or 14:30 UTC (depending on DST)
    # If UTC, 4:00 PM ET = 20:00 or 21:00 UTC
    
    morning_signals = signals[(signals['hour'] >= 13) & (signals['hour'] <= 15)]
    evening_signals = signals[(signals['hour'] >= 19) & (signals['hour'] <= 21)]
    
    print(f"Signals between 13:00-15:59: {len(morning_signals)}")
    print(f"Signals between 19:00-21:59: {len(evening_signals)}")
    
else:
    print(f"❌ File not found: {parquet_file}")

# Also check the market data
print("\n" + "="*60)
print("🏦 Checking market data timezone...")

market_data_paths = [
    Path('data/SPY_5m.parquet'),
    Path('data/SPY_5m.csv')
]

for path in market_data_paths:
    if path.exists():
        print(f"\nChecking: {path}")
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        
        # Try different timestamp column names
        ts_col = None
        for col in ['timestamp', 'Timestamp', 'datetime', 'Date', 'ts']:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col])
            print(f"First timestamp: {df[ts_col].iloc[0]}")
            print(f"Last timestamp: {df[ts_col].iloc[-1]}")
            
            # Check a specific date's range
            df['date'] = df[ts_col].dt.date
            sample_date = df['date'].iloc[len(df)//2]  # Middle of dataset
            sample_day = df[df['date'] == sample_date]
            print(f"\nSample day {sample_date}:")
            print(f"  First bar: {sample_day[ts_col].iloc[0]}")
            print(f"  Last bar: {sample_day[ts_col].iloc[-1]}")
        
        break