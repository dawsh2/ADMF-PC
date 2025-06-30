#!/usr/bin/env python3
"""Analyze exit types from position close events."""

import pandas as pd
import sys
import json
from pathlib import Path

# Get the latest results directory
if len(sys.argv) > 1:
    results_dir = Path(sys.argv[1])
else:
    # Find most recent results
    results_base = Path("config/bollinger/results")
    if results_base.exists():
        results_dir = max(results_base.iterdir(), key=lambda p: p.stat().st_mtime)
    else:
        print("No results directory found")
        sys.exit(1)

print(f"Analyzing results from: {results_dir}")

# Load position close events
positions_close_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"

if not positions_close_file.exists():
    print(f"No positions_close file found at {positions_close_file}")
    sys.exit(1)

# Read the parquet file
df = pd.read_parquet(positions_close_file)

print("\n=== Risk Management Exit Analysis ===")
print(f"Total positions closed: {len(df)}")

# Extract exit types from payload
exit_types = []
exit_reasons = []
highest_prices = []
entry_prices = []

for _, row in df.iterrows():
    metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
    exit_type = metadata.get('exit_type', 'None')
    exit_reason = metadata.get('exit_reason', 'None')
    exit_types.append(exit_type)
    exit_reasons.append(exit_reason)
    
    # Track highest prices for trailing stop analysis
    if 'highest_price' in metadata and 'entry_price' in metadata:
        highest_prices.append(float(metadata['highest_price']))
        entry_prices.append(float(metadata['entry_price']))

# Count exit types
from collections import Counter
exit_type_counts = Counter(exit_types)

print("\nExit types:")
for exit_type, count in exit_type_counts.most_common():
    percentage = (count / len(df)) * 100
    print(f"  {exit_type}: {count} ({percentage:.1f}%)")

# Show sample exit reasons
print("\nSample exit reasons:")
unique_reasons = list(set(exit_reasons))[:5]
for reason in unique_reasons:
    print(f"  {reason}")

# Calculate exit price accuracy
if len(df) > 0:
    first_metadata = json.loads(df.iloc[0]['metadata']) if isinstance(df.iloc[0]['metadata'], str) else df.iloc[0]['metadata']
    if 'exit_price' in first_metadata:
        print("\n=== Exit Price Analysis ===")
        
        actual_stops = 0
        for _, row in df.iterrows():
            metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
            if metadata.get('exit_type') == 'stop_loss':
                entry_price = metadata.get('entry_price', 0)
                exit_price = metadata.get('exit_price', 0)
                if entry_price > 0:
                    loss_pct = (exit_price - entry_price) / entry_price
                    print(f"  Stop loss: entry={entry_price:.4f}, exit={exit_price:.4f}, loss={loss_pct:.4%}")
                    # Check if close to exactly -0.1% or -0.2% (the two possible configured values)
                    if abs(abs(loss_pct) - 0.001) < 0.00005 or abs(abs(loss_pct) - 0.002) < 0.00005:
                        actual_stops += 1
    
    total_stops = exit_type_counts.get('stop_loss', 0)
    if total_stops > 0:
        accuracy = (actual_stops / total_stops) * 100
        print(f"Stop losses at exact -0.1%: {actual_stops}/{total_stops} ({accuracy:.1f}%)")

# Analyze trailing stop potential
if highest_prices and entry_prices:
    print("\n=== Trailing Stop Analysis ===")
    max_gains = [(h - e) / e for h, e in zip(highest_prices, entry_prices)]
    profitable_trades = sum(1 for g in max_gains if g > 0)
    
    print(f"Trades that went into profit: {profitable_trades}/{len(max_gains)} ({profitable_trades/len(max_gains)*100:.1f}%)")
    
    if profitable_trades > 0:
        profitable_gains = [g for g in max_gains if g > 0]
        print(f"Average max gain when profitable: {sum(profitable_gains)/len(profitable_gains)*100:.2f}%")
        print(f"Max gain achieved: {max(profitable_gains)*100:.2f}%")
        
        # Check how many would have hit trailing stop at different levels
        trailing_levels = [0.0001, 0.0003, 0.0005, 0.001, 0.002]
        print("\nPotential trailing stop hits at different levels:")
        for level in trailing_levels:
            hits = sum(1 for g in profitable_gains if g > level)
            print(f"  {level*100:.2f}%: {hits} trades ({hits/len(max_gains)*100:.1f}% of all trades)")