#!/usr/bin/env python3
import pandas as pd

# Read positions data
positions_open = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/20250627_173309/traces/portfolio/positions_open/positions_open.parquet')
positions_close = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/20250627_173309/traces/portfolio/positions_close/positions_close.parquet')

# Read signals for price data
signals = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/20250627_173309/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')

print("=== Checking Immediate Exit Examples ===\n")

# Find trades that exited within 1 bar
immediate_exits = []
for i in range(min(len(positions_open), len(positions_close))):
    open_bar = positions_open.iloc[i]['idx']
    close_bar = positions_close.iloc[i]['idx']
    
    if close_bar - open_bar <= 1:
        immediate_exits.append(i)

print(f"Found {len(immediate_exits)} trades that exited within 1 bar\n")

# Examine first few
for idx in immediate_exits[:5]:
    open_data = positions_open.iloc[idx]
    close_data = positions_close.iloc[idx]
    
    open_meta = open_data['metadata']
    close_meta = close_data['metadata']
    
    print(f"Trade {idx + 1}:")
    print(f"  Open bar: {open_data['idx']}")
    print(f"  Close bar: {close_data['idx']}")
    print(f"  Bars held: {close_data['idx'] - open_data['idx']}")
    
    if isinstance(open_meta, dict) and isinstance(close_meta, dict):
        qty = open_meta.get('quantity', 0)
        entry_price = open_meta.get('entry_price', 0)
        exit_price = close_meta.get('exit_price', 0)
        exit_type = close_meta.get('metadata', {}).get('exit_type', 'unknown')
        
        print(f"  Type: {'LONG' if qty > 0 else 'SHORT'}")
        print(f"  Entry price: ${entry_price:.2f}")
        print(f"  Exit price: ${exit_price:.2f}")
        print(f"  Exit type: {exit_type}")
        
        # Calculate what the stop price should be
        if exit_type == 'stop_loss':
            if qty > 0:  # Long
                expected_stop = entry_price * (1 - 0.00075)
                print(f"  Expected stop: ${expected_stop:.2f}")
                print(f"  Actual exit: ${exit_price:.2f}")
                print(f"  Difference: ${exit_price - expected_stop:.2f}")
            else:  # Short
                expected_stop = entry_price * (1 + 0.00075)
                print(f"  Expected stop: ${expected_stop:.2f}")
                print(f"  Actual exit: ${exit_price:.2f}")
                print(f"  Difference: ${exit_price - expected_stop:.2f}")
        
        # Try to find the bar's OHLC data
        signal_at_entry = signals[signals['idx'] == open_data['idx']]
        if len(signal_at_entry) > 0:
            sig_meta = signal_at_entry.iloc[0]['metadata']
            if isinstance(sig_meta, dict):
                print(f"  Entry bar OHLC: O=${sig_meta.get('open', 0):.2f}, H=${sig_meta.get('high', 0):.2f}, L=${sig_meta.get('low', 0):.2f}, C=${sig_meta.get('close', 0):.2f}")
        
        # Check next bar
        next_bar_signals = signals[signals['idx'] == close_data['idx']]
        if len(next_bar_signals) > 0:
            sig_meta = next_bar_signals.iloc[0]['metadata']
            if isinstance(sig_meta, dict):
                print(f"  Exit bar OHLC: O=${sig_meta.get('open', 0):.2f}, H=${sig_meta.get('high', 0):.2f}, L=${sig_meta.get('low', 0):.2f}, C=${sig_meta.get('close', 0):.2f}")
    
    print()

print("\n=== Analysis ===")
print("If many trades are hitting stops on the next bar, possible causes:")
print("1. Entry at close but stop checked against next bar's high/low")
print("2. Volatile market with gaps")
print("3. Stop loss too tight for the volatility")