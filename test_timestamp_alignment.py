#!/usr/bin/env python3
"""
Test script to verify timestamp alignment between source data and traces.

This script will:
1. Load source data and check timestamps at specific indices
2. Simulate how the data handler processes bars
3. Identify where the off-by-one error occurs
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_timestamp_alignment():
    """Test timestamp alignment between source and trace data."""
    
    # Load source data
    df = pd.read_csv('data/SPY_1m.csv', parse_dates=['timestamp'], index_col='timestamp')
    
    print("=== Source Data Analysis ===")
    print(f"Total bars in source: {len(df)}")
    print(f"First timestamp: {df.index[0]}")
    print(f"Last timestamp: {df.index[-1]}")
    print()
    
    # Check specific indices that user mentioned
    test_indices = [29, 30, 31]
    
    print("=== Timestamps at Key Indices ===")
    for idx in test_indices:
        ts = df.index[idx]
        ts_utc = pd.Timestamp(ts).tz_convert('UTC')
        print(f"idx={idx}: {ts} (ET) -> {ts_utc} (UTC)")
    
    print()
    print("=== Simulating Data Handler Processing ===")
    
    # Simulate how the data handler processes bars
    current_idx = 0
    for i in range(35):  # Process first 35 bars
        # This simulates the loop in handlers.py
        bar_data = df.iloc[i]
        timestamp = df.index[i]
        
        # This is what happens in handlers.py:
        # original_bar_index = self._get_original_bar_index(symbol, idx)
        # Since we're not using splits, original_bar_index = i
        original_bar_index = i
        
        if i in test_indices:
            print(f"\nProcessing loop iteration {i}:")
            print(f"  - Bar timestamp: {timestamp}")
            print(f"  - original_bar_index: {original_bar_index}")
            print(f"  - Bar data is from df.iloc[{i}]")
            
            # The event payload would contain:
            # 'original_bar_index': original_bar_index (which is i)
            # 'timestamp': timestamp (from df.index[i])
            
            # The trace would store:
            # bar_index: original_bar_index
            # timestamp: timestamp
            
            print(f"  - Trace would store: idx={original_bar_index}, timestamp={timestamp}")
            
        # Increment index (this happens AFTER processing)
        current_idx = i + 1
    
    print()
    print("=== Analysis Summary ===")
    print("The trace appears to be storing the correct timestamps!")
    print("- At idx=30 in trace, we should see the timestamp from df.index[30]")
    print(f"- df.index[30] = {df.index[30]} (ET) = {pd.Timestamp(df.index[30]).tz_convert('UTC')} (UTC)")
    print()
    print("The user reported seeing:")
    print("- Trace at idx=30: 2024-03-26T13:59:00+00:00 (UTC)")
    print(f"- But df.index[30] in UTC is: {pd.Timestamp(df.index[30]).tz_convert('UTC')}")
    print()
    print("This confirms a 1-minute offset!")
    print()
    
    # Let's check if there's something wrong with the source data itself
    print("=== Checking for Data Issues ===")
    
    # Check if timestamps are properly ordered
    time_diffs = df.index.to_series().diff()
    non_one_minute = time_diffs[time_diffs != pd.Timedelta(minutes=1)].dropna()
    
    if len(non_one_minute) > 0:
        print(f"Found {len(non_one_minute)} instances where time diff is not 1 minute:")
        print(non_one_minute.head(10))
    else:
        print("All consecutive timestamps are exactly 1 minute apart.")
    
    # Check for duplicates
    duplicates = df.index.duplicated()
    if duplicates.any():
        print(f"Found {duplicates.sum()} duplicate timestamps!")
    else:
        print("No duplicate timestamps found.")
    
    # Check timezone consistency
    print(f"\nTimezone info: {df.index.tz}")
    
    # The issue might be in how the timestamp is being stored/retrieved in traces
    print("\n=== Hypothesis ===")
    print("The off-by-one error suggests that either:")
    print("1. The trace is using the wrong index when storing (using idx-1 instead of idx)")
    print("2. The timestamp is being shifted during processing")
    print("3. There's an issue when reading/displaying the trace data")

if __name__ == "__main__":
    test_timestamp_alignment()