#!/usr/bin/env python3
"""
Analyze if the notebook uses a different entry/exit methodology
"""
import pandas as pd
import numpy as np

# Read signals
signals = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/latest/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')

print("=== Analyzing Entry Points ===\n")

# The notebook might be doing one of these:
# 1. Enter at CLOSE of signal bar, exit based on NEXT bar's high/low
# 2. Enter at OPEN of next bar, exit based on SAME bar's high/low
# 3. Enter at a better price within the bar (e.g., near the bands)

# Let's check some signal bars to understand the price action
signal_changes = []
for i in range(1, len(signals)):
    prev_val = signals.iloc[i-1]['val']
    curr_val = signals.iloc[i]['val']
    
    if prev_val == 0 and curr_val != 0:  # Entry signal
        signal_changes.append({
            'idx': signals.iloc[i]['idx'],
            'direction': 'LONG' if curr_val > 0 else 'SHORT',
            'metadata': signals.iloc[i]['metadata']
        })

print(f"Found {len(signal_changes)} entry signals\n")

# Analyze first few entries
print("=== Entry Signal Analysis ===")
for i, entry in enumerate(signal_changes[:5]):
    meta = entry['metadata']
    if isinstance(meta, dict):
        print(f"\nSignal {i+1} - {entry['direction']}:")
        print(f"  Bar index: {entry['idx']}")
        print(f"  Open: ${meta.get('open', 0):.2f}")
        print(f"  High: ${meta.get('high', 0):.2f}")
        print(f"  Low: ${meta.get('low', 0):.2f}")
        print(f"  Close: ${meta.get('close', 0):.2f}")
        print(f"  Range: ${meta.get('high', 0) - meta.get('low', 0):.2f}")
        
        # Calculate where stops would be from different entry points
        stop_pct = 0.00075
        if entry['direction'] == 'LONG':
            stop_from_close = meta.get('close', 0) * (1 - stop_pct)
            stop_from_open = meta.get('open', 0) * (1 - stop_pct)
            print(f"  Stop from close: ${stop_from_close:.2f}")
            print(f"  Stop from open: ${stop_from_open:.2f}")
            print(f"  Next bar needs low > ${stop_from_close:.2f} to survive")
        else:  # SHORT
            stop_from_close = meta.get('close', 0) * (1 + stop_pct)
            stop_from_open = meta.get('open', 0) * (1 + stop_pct)
            print(f"  Stop from close: ${stop_from_close:.2f}")
            print(f"  Stop from open: ${stop_from_open:.2f}")
            print(f"  Next bar needs high < ${stop_from_close:.2f} to survive")

print("\n=== Key Insight ===")
print("If we're entering at CLOSE price of signal bar and checking stops against")
print("the NEXT bar's high/low, we're much more vulnerable to being stopped out")
print("than if we entered at OPEN of the next bar.")
print("\nThe notebook might be:")
print("1. Entering at a more favorable price (not just close)")
print("2. Using a different bar for stop calculation")
print("3. Applying stops differently (e.g., end-of-bar check vs intrabar)")

# Check typical bar ranges vs our stop
print("\n=== Bar Range Analysis ===")
ranges = []
for i in range(len(signals)):
    meta = signals.iloc[i]['metadata']
    if isinstance(meta, dict):
        high = meta.get('high', 0)
        low = meta.get('low', 0)
        close = meta.get('close', 0)
        if high > 0 and low > 0 and close > 0:
            range_pct = (high - low) / close * 100
            ranges.append(range_pct)

if ranges:
    print(f"Average bar range: {np.mean(ranges):.3f}%")
    print(f"Median bar range: {np.median(ranges):.3f}%")
    print(f"Stop loss: 0.075%")
    print(f"Bars with range > stop: {sum(1 for r in ranges if r > 0.075)} ({sum(1 for r in ranges if r > 0.075)/len(ranges)*100:.1f}%)")
    
    # This is the key insight
    if np.median(ranges) > 0.075:
        print("\n⚠️ CRITICAL: Most bars have a range larger than our stop loss!")
        print("This means if we enter at close and check against next bar's full range,")
        print("we'll likely get stopped out immediately on normal market movement.")