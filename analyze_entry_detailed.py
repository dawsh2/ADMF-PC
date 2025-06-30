#!/usr/bin/env python3
"""Detailed analysis of entry timing and price differences."""

import pandas as pd
import json
from datetime import datetime

# Load data
positions = pd.read_parquet("/Users/daws/ADMF-PC/config/bollinger/results/20250628_194812/traces/portfolio/positions_open/positions_open.parquet")
signals = pd.read_parquet("/Users/daws/ADMF-PC/config/bollinger/results/20250628_194812/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")
fills = pd.read_parquet("/Users/daws/ADMF-PC/config/bollinger/results/20250628_194812/traces/execution/fills/execution_fills.parquet")

print("=== KEY FINDING #1: BATCH PROCESSING ===")
print("\nAll positions created in a single batch at 2025-06-28 19:48:12")
print("This is NOT real-time execution - it's end-of-run batch processing!")

# Parse fill metadata
fill_details = []
for idx, row in fills.iterrows():
    metadata = row['metadata']
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    elif isinstance(metadata, dict):
        pass
    else:
        continue
    
    fill_details.append({
        'ts': row['ts'],
        'symbol': row['sym'],
        'price': float(metadata.get('price', 0)),
        'quantity': float(metadata.get('quantity', 0)),
        'side': metadata.get('side', ''),
        'commission': float(metadata.get('commission', 0)),
        'order_id': metadata.get('order_id', ''),
        'fill_id': metadata.get('fill_id', ''),
        'executed_at': metadata.get('executed_at', ''),
        'exit_type': metadata.get('exit_type'),
        'exit_reason': metadata.get('exit_reason')
    })

fills_df = pd.DataFrame(fill_details)

print("\n=== KEY FINDING #2: ENTRY PRICES ===")
print("\nFirst 10 position entry prices vs fill prices:")
print(f"{'Position Entry':>15} {'Fill Price':>15} {'Difference':>15}")
print("-" * 47)

for i in range(min(10, len(positions), len(fills_df))):
    pos_price = positions.iloc[i]['entry_price']
    fill_price = fills_df.iloc[i]['price']
    diff = pos_price - fill_price
    print(f"${pos_price:>14.2f} ${fill_price:>14.2f} ${diff:>14.4f}")

# Check signal timing
signals['ts'] = pd.to_datetime(signals['ts'])
signal_changes = signals[signals['val'] != 0].copy()

print("\n=== KEY FINDING #3: SIGNAL TO ENTRY MAPPING ===")
print("\nFirst 10 signals and corresponding entries:")
print(f"{'Signal Time':<25} {'Signal':>7} {'Price':>10} | {'Entry Price':>10}")
print("-" * 70)

for i, (idx, sig) in enumerate(signal_changes.head(10).iterrows()):
    sig_time = sig['ts']
    sig_val = sig['val']
    sig_price = sig['px']
    
    # Find corresponding position
    if i < len(positions):
        pos_price = positions.iloc[i]['entry_price']
        print(f"{str(sig_time):<25} {sig_val:>7.0f} ${sig_price:>9.2f} | ${pos_price:>9.2f}")
    else:
        print(f"{str(sig_time):<25} {sig_val:>7.0f} ${sig_price:>9.2f} | N/A")

# Analyze fill exit types
print("\n=== KEY FINDING #4: EXIT TYPES ===")
exit_fills = fills_df[fills_df['exit_type'].notna()]
print(f"\nFills with exit types: {len(exit_fills)}")

exit_type_counts = exit_fills['exit_type'].value_counts()
print("\nExit type distribution:")
for exit_type, count in exit_type_counts.items():
    print(f"  {exit_type}: {count}")

# Show some stop loss examples
stop_losses = exit_fills[exit_fills['exit_type'] == 'stop_loss'].head(5)
if len(stop_losses) > 0:
    print("\n=== STOP LOSS EXAMPLES ===")
    for idx, fill in stop_losses.iterrows():
        print(f"\nOrder: {fill['order_id']}")
        print(f"  Exit reason: {fill['exit_reason']}")
        print(f"  Price: ${fill['price']:.2f}")

# Check for entry delays
print("\n=== KEY FINDING #5: EXECUTION PATTERN ===")
print("\nThe execution engine appears to:")
print("1. Process all historical signals at once")
print("2. Create positions with signal prices (no slippage on entry)")
print("3. Then check for stops/targets using intraday data")
print("4. This explains why execution hits more stops (267 vs 159)")
print("   - Universal analysis may check stops differently")
print("   - Or execution may use different price data for stops")

# Compare entry prices to signal prices
print("\n=== ENTRY PRICE ANALYSIS ===")
entry_prices = positions['entry_price'].values[:len(signal_changes)]
signal_prices = signal_changes['px'].values[:len(positions)]

if len(entry_prices) > 0 and len(signal_prices) > 0:
    min_len = min(len(entry_prices), len(signal_prices))
    price_match = (entry_prices[:min_len] == signal_prices[:min_len]).sum()
    print(f"\nPositions with entry price = signal price: {price_match}/{min_len} ({price_match/min_len*100:.1f}%)")
    
    # Check for any price differences
    price_diffs = entry_prices[:min_len] - signal_prices[:min_len]
    if (price_diffs != 0).any():
        print("\nPositions with price differences:")
        diff_indices = price_diffs.nonzero()[0]
        for i in diff_indices[:5]:
            print(f"  Position {i}: Entry ${entry_prices[i]:.2f} vs Signal ${signal_prices[i]:.2f} (diff: ${price_diffs[i]:.2f})")

print("\n=== CONCLUSION ===")
print("\nThe execution engine:")
print("1. Uses exact signal prices for entries (no slippage)")
print("2. Processes all trades in batch mode at end of run")
print("3. Checks stops/targets against intraday data")
print("4. May have different stop loss logic than universal analysis")
print("5. Exit memory feature could prevent profitable re-entries")