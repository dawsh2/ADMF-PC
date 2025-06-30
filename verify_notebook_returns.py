#!/usr/bin/env python3
import pandas as pd

# Read our position close data
positions_close = pd.read_parquet('/Users/daws/ADMF-PC/config/bollinger/results/20250627_173309/traces/portfolio/positions_close/positions_close.parquet')

# Calculate returns the way the notebook does
trades = []
for i in range(len(positions_close)):
    metadata = positions_close.iloc[i]['metadata']
    if isinstance(metadata, dict):
        qty = metadata.get('quantity', 0)
        entry_price = metadata.get('entry_price', 0)
        exit_price = metadata.get('exit_price', 0)
        
        if entry_price > 0:
            # Calculate return based on position direction
            if qty > 0:  # Long
                return_pct = ((exit_price - entry_price) / entry_price) * 100
            else:  # Short
                return_pct = ((entry_price - exit_price) / entry_price) * 100
            
            trades.append(return_pct)

# Calculate metrics the notebook way
print("=== Notebook-style Return Calculation ===")
print(f"Total trades: {len(trades)}")
print(f"Average return per trade: {sum(trades)/len(trades):.2f}%")
print(f"Total return (sum): {sum(trades):.2f}%")

# Calculate win rate
wins = sum(1 for r in trades if r > 0)
win_rate = wins / len(trades) * 100
print(f"Win rate: {win_rate:.1f}%")

print("\n=== Comparison ===")
print(f"Notebook expects: ~10.27% total return (sum of all trades)")
print(f"We're getting: {sum(trades):.2f}% total return")
print(f"Ratio: {sum(trades) / 10.27:.2f}x")

# Check a few individual trades
print("\nFirst 10 trade returns:")
for i, ret in enumerate(trades[:10]):
    print(f"  Trade {i+1}: {ret:.3f}%")