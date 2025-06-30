#!/usr/bin/env python3
"""Analyze trades exiting at exactly 0.075% gain (stop loss value)."""

import pandas as pd
import numpy as np
from pathlib import Path

print("=== Analyzing Suspicious Exits at 0.075% Gain ===")

# Load trades
results_dir = Path("config/bollinger/results/latest")
trades_file = results_dir / "traces/events/portfolio/trades.parquet"

if not trades_file.exists():
    print(f"Error: Trades file not found at {trades_file}")
    exit(1)

trades = pd.read_parquet(trades_file)
print(f"Total trades: {len(trades)}")

# Calculate returns
trades['return_pct'] = ((trades['exit_price'] - trades['entry_price']) / trades['entry_price']) * 100

# Look for trades with returns very close to 0.075%
tolerance = 0.001  # 0.001% tolerance
suspicious_trades = trades[abs(trades['return_pct'] - 0.075) < tolerance]

print(f"\nTrades with ~0.075% return: {len(suspicious_trades)}")

if len(suspicious_trades) > 0:
    print("\nBreakdown by exit type:")
    exit_type_counts = suspicious_trades['exit_type'].value_counts()
    print(exit_type_counts)
    
    print("\nSample of suspicious trades:")
    for i, (_, trade) in enumerate(suspicious_trades.head(5).iterrows()):
        print(f"\n{i+1}. Trade at {trade['timestamp']}")
        print(f"   Entry: ${trade['entry_price']:.4f}")
        print(f"   Exit: ${trade['exit_price']:.4f}")
        print(f"   Return: {trade['return_pct']:.4f}%")
        print(f"   Exit type: {trade['exit_type']}")
        print(f"   Direction: {'LONG' if trade['quantity'] > 0 else 'SHORT'}")
    
    # Check if these are actually stop losses being triggered in the wrong direction
    long_trades = suspicious_trades[suspicious_trades['quantity'] > 0]
    short_trades = suspicious_trades[suspicious_trades['quantity'] < 0]
    
    print(f"\nLong trades with 0.075% gain: {len(long_trades)}")
    print(f"Short trades with 0.075% gain: {len(short_trades)}")
    
    # For long trades, stop loss should be a loss, not a gain
    if len(long_trades) > 0:
        print("\n⚠️ WARNING: Long trades exiting at +0.075% when stop loss should be -0.075%!")
        print("This suggests the stop loss logic might be inverted!")

# Also check for trades at -0.075% (where stop losses should be)
stop_loss_trades = trades[abs(trades['return_pct'] + 0.075) < tolerance]
print(f"\n\nTrades with ~-0.075% return: {len(stop_loss_trades)}")
if len(stop_loss_trades) > 0:
    print("Exit types:")
    print(stop_loss_trades['exit_type'].value_counts())

# Check take profit trades at 0.1%
take_profit_trades = trades[abs(trades['return_pct'] - 0.1) < tolerance]
print(f"\n\nTrades with ~0.1% return (take profit): {len(take_profit_trades)}")
if len(take_profit_trades) > 0:
    print("Exit types:")
    print(take_profit_trades['exit_type'].value_counts())

# Look for any pattern in exit prices
print("\n\n=== Exit Price Analysis ===")
trades['exit_price_ratio'] = trades['exit_price'] / trades['entry_price']

# Check common exit ratios
common_ratios = [0.99925, 1.00075, 0.999, 1.001]  # -0.075%, +0.075%, -0.1%, +0.1%
for ratio in common_ratios:
    matching = trades[abs(trades['exit_price_ratio'] - ratio) < 0.00001]
    if len(matching) > 0:
        pct = (ratio - 1) * 100
        print(f"\nTrades exiting at {pct:+.3f}%: {len(matching)}")
        print(f"  Exit types: {matching['exit_type'].value_counts().to_dict()}")