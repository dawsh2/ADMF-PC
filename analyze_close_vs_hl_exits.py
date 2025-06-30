#!/usr/bin/env python3
"""Analyze how many exits would be missed using close vs high/low."""

import pandas as pd

print("=== Close Price vs High/Low Exit Analysis ===")

# Load data
df = pd.read_parquet("data/spy_5m_full.parquet")

# Simulate some positions and check exits
stop_loss_pct = 0.00075
take_profit_pct = 0.001

close_exits = 0
hl_exits = 0
both_methods_exit = 0
only_hl_exits = 0
only_close_exits = 0

# Sample 1000 random "entries"
sample_indices = df.sample(min(1000, len(df)-1)).index

for idx in sample_indices:
    if idx >= len(df) - 1:
        continue
        
    # Simulate entry at this bar's close
    entry_bar = df.iloc[idx]
    entry_price = entry_bar['close']
    
    # Next bar is where we check for exits
    next_bar = df.iloc[idx + 1]
    
    # Calculate exit levels for long position
    sl_price = entry_price * (1 - stop_loss_pct)
    tp_price = entry_price * (1 + take_profit_pct)
    
    # Method 1: Close price only (old way)
    close_hit_sl = next_bar['close'] <= sl_price
    close_hit_tp = next_bar['close'] >= tp_price
    close_exit = close_hit_sl or close_hit_tp
    
    # Method 2: High/Low prices (new way)
    hl_hit_sl = next_bar['low'] <= sl_price
    hl_hit_tp = next_bar['high'] >= tp_price
    hl_exit = hl_hit_sl or hl_hit_tp
    
    if close_exit:
        close_exits += 1
    if hl_exit:
        hl_exits += 1
    if close_exit and hl_exit:
        both_methods_exit += 1
    if hl_exit and not close_exit:
        only_hl_exits += 1
    if close_exit and not hl_exit:
        only_close_exits += 1

print(f"\nResults from {len(sample_indices)} simulated positions:")
print(f"Exits using close price only: {close_exits}")
print(f"Exits using high/low prices: {hl_exits}")
print(f"Exits caught by both methods: {both_methods_exit}")
print(f"Exits ONLY caught by high/low: {only_hl_exits}")
print(f"Exits ONLY caught by close: {only_close_exits}")

if hl_exits > close_exits:
    increase_pct = ((hl_exits - close_exits) / close_exits * 100) if close_exits > 0 else 0
    print(f"\n⚠️ High/Low catches {increase_pct:.1f}% MORE exits than close price!")
    print("This explains why trades increased from 453 to 463.")

print("\n\n=== Implications ===")
print("The OHLC fix makes the backtest MORE ACCURATE but different from the notebook.")
print("The notebook likely uses close prices for exits (less accurate but simpler).")
print("\nTo match the notebook exactly, we'd need to:")
print("1. Revert to using close prices for exit checks (less accurate)")
print("2. OR accept that our results are more accurate but different")