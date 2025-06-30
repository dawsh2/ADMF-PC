#!/usr/bin/env python3
"""
Simple notebook snippet to check entry timing
Just paste this into a notebook cell and run it
"""

# First, show what variables you have available
print("=== Step 1: Check Available DataFrames ===")
print("Run this in a cell:")
print("%whos DataFrame")
print()

# Then check the structure
print("=== Step 2: Check Your Data Structure ===")
print("Replace 'your_df' with your actual DataFrame name and run:")
print("""
# Check columns
print(your_df.columns.tolist())
print()
# Check first few rows
print(your_df.head())
""")
print()

# The analysis function to paste
print("=== Step 3: Paste and Run This Analysis ===")
print("Copy everything below into a cell:")
print("-" * 60)

analysis_code = '''
import pandas as pd
import numpy as np

def check_entry_timing(df, signal_col='signal'):
    """Check if entering at CLOSE vs OPEN explains stop loss differences"""
    
    # Constants
    STOP_LOSS = 0.00075  # 0.075%
    
    # Find entries
    entries = []
    for i in range(1, len(df)-1):
        prev_sig = df.iloc[i-1][signal_col]
        curr_sig = df.iloc[i][signal_col]
        if prev_sig == 0 and curr_sig != 0:
            entries.append(i)
    
    print(f"Found {len(entries)} entry signals\\n")
    
    # Show first 3 examples
    for idx, i in enumerate(entries[:3]):
        signal_bar = df.iloc[i]
        next_bar = df.iloc[i+1]
        
        print(f"Entry {idx+1}:")
        print(f"  Signal bar close: ${signal_bar['close']:.2f}")
        print(f"  Next bar open: ${next_bar['open']:.2f}")
        print(f"  Next bar low: ${next_bar['low']:.2f}")
        
        # Check stops
        if signal_bar[signal_col] > 0:  # Long
            stop_close = signal_bar['close'] * (1 - STOP_LOSS)
            stop_open = next_bar['open'] * (1 - STOP_LOSS)
            print(f"  Stop from close: ${stop_close:.2f}")
            print(f"  Stop from open: ${stop_open:.2f}")
            print(f"  Would hit stop from close? {next_bar['low'] <= stop_close}")
            print(f"  Would hit stop from open? {next_bar['low'] <= stop_open}")
        print()
    
    # Count stops
    stops_close = 0
    stops_open = 0
    
    for i in entries:
        signal_bar = df.iloc[i]
        next_bar = df.iloc[i+1]
        
        if signal_bar[signal_col] > 0:  # Long
            stop_close = signal_bar['close'] * (1 - STOP_LOSS)
            stop_open = next_bar['open'] * (1 - STOP_LOSS)
            if next_bar['low'] <= stop_close:
                stops_close += 1
            if next_bar['low'] <= stop_open:
                stops_open += 1
        else:  # Short
            stop_close = signal_bar['close'] * (1 + STOP_LOSS)
            stop_open = next_bar['open'] * (1 + STOP_LOSS)
            if next_bar['high'] >= stop_close:
                stops_close += 1
            if next_bar['high'] >= stop_open:
                stops_open += 1
    
    print("\\n=== SUMMARY ===")
    print(f"Stops hit on next bar:")
    print(f"  Entering at CLOSE: {stops_close}/{len(entries)} ({stops_close/len(entries)*100:.1f}%)")
    print(f"  Entering at OPEN:  {stops_open}/{len(entries)} ({stops_open/len(entries)*100:.1f}%)")
    print(f"\\nDifference: {stops_close - stops_open} fewer stops with OPEN entry")

# Run the analysis - adjust these based on your DataFrame
# Example 1: If your DataFrame is called 'df' with a 'signal' column:
# check_entry_timing(df, 'signal')

# Example 2: If your DataFrame has different names:
# check_entry_timing(data, 'bb_signal')

# Example 3: Show me your columns first:
# print("Columns:", df.columns.tolist())
'''

print(analysis_code)
print("-" * 60)

print("\n=== Step 4: Share Results ===")
print("After running, share:")
print("1. Your DataFrame columns")
print("2. The output showing stop percentages")
print("3. Any error messages")