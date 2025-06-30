#!/usr/bin/env python3
"""Check what price data is included in signals."""

import pandas as pd
from pathlib import Path
import json

print("=== Checking Signal Price Data ===")

# Check a signal event to see what price data is included
results_dir = Path("config/bollinger/results/latest")
signal_events_file = results_dir / "traces/events/strategy/signal_events.parquet"

if signal_events_file.exists():
    signals = pd.read_parquet(signal_events_file)
    print(f"Found {len(signals)} signal events")
    
    # Look at first few signals
    print("\nFirst signal structure:")
    first_signal = signals.iloc[0]
    for key, value in first_signal.items():
        print(f"  {key}: {value}")
    
    # Check if we have OHLC data
    print("\n\nChecking for OHLC data in signals:")
    has_open = 'open' in signals.columns or (signals['metadata'].iloc[0] and 'open' in signals['metadata'].iloc[0])
    has_high = 'high' in signals.columns or (signals['metadata'].iloc[0] and 'high' in signals['metadata'].iloc[0])
    has_low = 'low' in signals.columns or (signals['metadata'].iloc[0] and 'low' in signals['metadata'].iloc[0])
    has_close = 'close' in signals.columns or (signals['metadata'].iloc[0] and 'close' in signals['metadata'].iloc[0])
    
    print(f"  Has open: {has_open}")
    print(f"  Has high: {has_high}")
    print(f"  Has low: {has_low}")
    print(f"  Has close: {has_close}")
    
    # Check what price field exists
    if 'price' in signals.columns:
        print(f"\n'price' field exists - sample values: {signals['price'].head()}")
    
    print("\n\n⚠️ CRITICAL ISSUE IDENTIFIED:")
    print("If signals only contain close price, the risk manager CANNOT:")
    print("1. Calculate accurate intrabar stop loss prices (needs low)")
    print("2. Calculate accurate intrabar take profit prices (needs high)")
    print("3. This explains why exits might be happening at wrong prices!")

# Let's also check what the strategy is sending
print("\n\n=== Checking Strategy Signal Generation ===")
strategy_file = Path("src/strategy/strategies/indicators/crossovers.py")
if strategy_file.exists():
    with open(strategy_file, 'r') as f:
        content = f.read()
        if "SignalEvent" in content and "price" in content:
            print("Strategy appears to include price in signals")
            # Find the signal generation code
            import re
            signal_pattern = r'SignalEvent\([^)]+\)'
            matches = re.findall(signal_pattern, content)
            if matches:
                print("\nSample signal generation code:")
                for match in matches[:2]:
                    print(f"  {match}")
else:
    print(f"Strategy file not found: {strategy_file}")