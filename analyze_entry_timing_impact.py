#!/usr/bin/env python3
"""
Analyze the impact of entering at CLOSE vs OPEN of next bar
"""
import pandas as pd
import numpy as np

# Read data
signals = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/latest/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')

print("=== Entry Timing Impact Analysis ===\n")

# Find all entry signals
entry_signals = []
for i in range(1, len(signals)-1):
    prev_val = signals.iloc[i-1]['val']
    curr_val = signals.iloc[i]['val']
    
    if prev_val == 0 and curr_val != 0:  # Entry signal
        entry_signals.append(i)

print(f"Found {len(entry_signals)} entry signals\n")

# Analyze what would happen with different entry methods
stop_loss_pct = 0.00075
close_entry_stops = 0
open_entry_stops = 0

for idx in entry_signals[:100]:  # Check first 100 entries
    signal_bar = signals.iloc[idx]
    next_bar = signals.iloc[idx + 1] if idx + 1 < len(signals) else None
    
    if next_bar is None:
        continue
    
    signal_meta = signal_bar['metadata']
    next_meta = next_bar['metadata']
    
    if isinstance(signal_meta, dict) and isinstance(next_meta, dict):
        # Method 1: Enter at CLOSE of signal bar
        close_entry = signal_meta.get('close', 0)
        
        # Method 2: Enter at OPEN of next bar
        open_entry = next_meta.get('open', 0)
        
        # Check stops against next bar's range
        next_high = next_meta.get('high', 0)
        next_low = next_meta.get('low', 0)
        
        if signal_bar['val'] > 0:  # Long signal
            # Calculate stop prices
            close_stop = close_entry * (1 - stop_loss_pct)
            open_stop = open_entry * (1 - stop_loss_pct)
            
            # Check if stopped out
            if next_low <= close_stop:
                close_entry_stops += 1
            if next_low <= open_stop:
                open_entry_stops += 1
                
        else:  # Short signal
            # Calculate stop prices
            close_stop = close_entry * (1 + stop_loss_pct)
            open_stop = open_entry * (1 + stop_loss_pct)
            
            # Check if stopped out
            if next_high >= close_stop:
                close_entry_stops += 1
            if next_high >= open_stop:
                open_entry_stops += 1

print(f"Stop losses hit on next bar:")
print(f"  Entering at CLOSE: {close_entry_stops} ({close_entry_stops/len(entry_signals[:100])*100:.1f}%)")
print(f"  Entering at OPEN:  {open_entry_stops} ({open_entry_stops/len(entry_signals[:100])*100:.1f}%)")
print(f"  Difference: {close_entry_stops - open_entry_stops} fewer stops with open entry")

print("\n=== Gap Analysis ===")
# Check overnight gaps which could make this worse
gaps = []
for i in range(1, len(signals)):
    curr_meta = signals.iloc[i]['metadata']
    prev_meta = signals.iloc[i-1]['metadata']
    
    if isinstance(curr_meta, dict) and isinstance(prev_meta, dict):
        curr_open = curr_meta.get('open', 0)
        prev_close = prev_meta.get('close', 0)
        
        if prev_close > 0:
            gap_pct = abs(curr_open - prev_close) / prev_close * 100
            gaps.append(gap_pct)

if gaps:
    print(f"Average gap (open vs prev close): {np.mean(gaps):.3f}%")
    print(f"Gaps > stop loss ({stop_loss_pct*100:.3f}%): {sum(1 for g in gaps if g > stop_loss_pct*100)} "
          f"({sum(1 for g in gaps if g > stop_loss_pct*100)/len(gaps)*100:.1f}%)")

print("\n=== Recommendation ===")
print("The notebook likely enters at the OPEN of the bar AFTER the signal,")
print("not at the CLOSE of the signal bar. This would explain why they have")
print("fewer stop losses (20.7% vs our 44.6%).")
print("\nThis is more realistic as you can't know a signal until the bar closes,")
print("so you'd enter at the next bar's open price.")