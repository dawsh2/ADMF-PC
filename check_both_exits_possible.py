#!/usr/bin/env python3
"""Check if both SL and TP could be hit in same bar."""

import pandas as pd
from decimal import Decimal

print("=== Checking if Both Exits Possible in Same Bar ===")

# Example calculation
entry_price = 100.0
stop_loss_pct = 0.00075  # 0.075%
take_profit_pct = 0.001  # 0.1%

# For a long position
sl_price = entry_price * (1 - stop_loss_pct)  # 99.925
tp_price = entry_price * (1 + take_profit_pct)  # 100.1

print(f"Long position at ${entry_price}:")
print(f"  Stop loss at: ${sl_price:.3f} (-0.075%)")
print(f"  Take profit at: ${tp_price:.3f} (+0.1%)")
print(f"  Range needed: ${tp_price - sl_price:.3f} ({(tp_price - sl_price)/entry_price*100:.3f}%)")

# Load data to check
df = pd.read_parquet("data/spy_5m_full.parquet")

# For each bar, check if it could trigger both
both_possible = 0
for _, bar in df.iterrows():
    # Assume entry at open
    entry = bar['open']
    if entry <= 0:
        continue
        
    # Calculate exit levels
    sl_long = entry * (1 - stop_loss_pct)
    tp_long = entry * (1 + take_profit_pct)
    
    # Check if bar range encompasses both
    if bar['low'] <= sl_long and bar['high'] >= tp_long:
        both_possible += 1

print(f"\n\nBars where BOTH exits could trigger: {both_possible} ({both_possible/len(df)*100:.2f}%)")

# Let's also check the notebook's approach
print("\n\n=== Notebook Approach ===")
print("The notebook likely uses a specific order of operations:")
print("1. Check stop loss first (more urgent)")
print("2. Only check take profit if stop loss didn't trigger")
print("3. This prevents double-exits on the same bar")

print("\n\n=== Current Implementation Issue ===")
print("The risk manager checks SL and TP independently.")
print("If a bar can hit both, it might be creating TWO exit signals!")
print("This would increase trade count as positions exit and re-enter more often.")