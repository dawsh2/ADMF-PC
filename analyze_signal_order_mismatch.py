#!/usr/bin/env python3
"""Analyze why we have 9 signals but only 8 orders."""

import pandas as pd
import json
from pathlib import Path

workspace = Path("config/bollinger/results/latest")

# Load signals
signals = pd.read_parquet(workspace / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
print("=== Signal Sequence ===")
print(f"Total signals: {len(signals)}")
for i, (_, signal) in enumerate(signals.iterrows()):
    print(f"{i+1}. Time: {signal['ts']}, Value: {signal['val']}, Price: ${signal['px']:.2f}")

# Load orders
orders = pd.read_parquet(workspace / "traces/portfolio/orders/portfolio_orders.parquet")
print(f"\n=== Orders Generated ===")
print(f"Total orders: {len(orders)}")
for i, (_, order) in enumerate(orders.iterrows()):
    order_data = json.loads(order['metadata'])
    print(f"{i+1}. Time: {order['ts']}, Side: {order_data['side']}, Price: ${order_data['price']}")

# Load positions to understand the flow
positions_open = pd.read_parquet(workspace / "traces/portfolio/positions_open/positions_open.parquet")
positions_close = pd.read_parquet(workspace / "traces/portfolio/positions_close/positions_close.parquet")

print(f"\n=== Position Events ===")
print(f"Opens: {len(positions_open)}, Closes: {len(positions_close)}")

# Analyze the signal pattern
print("\n=== Signal Pattern Analysis ===")
signal_values = signals['val'].tolist()
print(f"Signal sequence: {signal_values}")
print(f"Signal changes: {[signal_values[i] for i in range(len(signal_values)) if i == 0 or signal_values[i] != signal_values[i-1]]}")

# Count transitions
transitions = []
for i in range(1, len(signal_values)):
    if signal_values[i] != signal_values[i-1]:
        transitions.append(f"{signal_values[i-1]} → {signal_values[i]}")

print(f"\nTransitions: {transitions}")
print(f"Number of transitions: {len(transitions)}")

# Check for duplicate signals or signals that might be ignored
print("\n=== Possible Issues ===")
if signal_values[0] == 0:
    print("- First signal is 0 (FLAT) - no position to close, so no order generated")
if signal_values[-1] != 0:
    print("- Last signal is not 0 - position may still be open")

# Check for consecutive same signals
for i in range(1, len(signal_values)):
    if signal_values[i] == signal_values[i-1] and signal_values[i] != 0:
        print(f"- Duplicate signal at index {i}: {signal_values[i]} (would be ignored)")