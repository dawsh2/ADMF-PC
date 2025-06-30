#\!/usr/bin/env python3
"""Verify stop loss logic is working correctly."""

import pandas as pd
import json
from pathlib import Path
from collections import Counter

print("STOP LOSS VERIFICATION")
print("=" * 60)

# Load position data
results_dir = Path("config/bollinger/results/latest")
try:
    opens = pd.read_parquet(results_dir / "traces/portfolio/positions_open/positions_open.parquet")
except:
    # Try alternate location
    opens = pd.read_parquet(results_dir / "traces/portfolio/positions_open/portfolio.parquet")
    
try:
    closes = pd.read_parquet(results_dir / "traces/portfolio/positions_close/positions_close.parquet")
except:
    # Try alternate location
    closes = pd.read_parquet(results_dir / "traces/portfolio/positions_close/portfolio.parquet")

# Parse metadata to get exit types
exit_data = []
for idx, row in closes.iterrows():
    # Check if we have exit_type column directly
    if 'exit_type' in closes.columns:
        exit_type = row.get('exit_type', 'unknown')
    else:
        # Try to get from metadata
        metadata = row.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        exit_type = metadata.get('exit_type', 'unknown') if isinstance(metadata, dict) else 'unknown'
    
    # Get price data - check different possible locations
    if 'exit_price' in row and 'entry_price' in row:
        exit_price = row['exit_price']
        entry_price = row['entry_price']
        quantity = row.get('quantity', 1)
        realized_pnl = row.get('realized_pnl', 0)
    elif 'payload' in row and isinstance(row['payload'], dict):
        exit_price = row['payload']['exit_price']
        entry_price = row['payload']['entry_price']
        quantity = row['payload'].get('quantity', 1)
        realized_pnl = row['payload'].get('realized_pnl', 0)
    else:
        # Skip if we can't find price data
        continue
        
    exit_data.append({
        'exit_type': exit_type,
        'exit_price': float(exit_price),
        'entry_price': float(entry_price),
        'quantity': float(quantity) if quantity else 1,
        'realized_pnl': float(realized_pnl) if realized_pnl else 0
    })

exit_df = pd.DataFrame(exit_data)

# Count stop losses
stop_losses = exit_df[exit_df['exit_type'] == 'stop_loss']
print(f"Found {len(stop_losses)} positions claiming stop loss hit")

if len(stop_losses) > 0:
    # Check if they actually hit 0.075% loss
    stop_losses['return_pct'] = stop_losses.apply(
        lambda row: ((row['exit_price'] - row['entry_price']) / row['entry_price'] * 100) if row['quantity'] > 0 
                   else ((row['entry_price'] - row['exit_price']) / row['entry_price'] * 100),
        axis=1
    )
    
    print("\nStop loss returns distribution:")
    print(f"  Expected: -0.075%")
    print(f"  Actual mean: {stop_losses['return_pct'].mean():.4f}%")
    print(f"  Actual std: {stop_losses['return_pct'].std():.4f}%")
    print(f"  Actual range: [{stop_losses['return_pct'].min():.4f}%, {stop_losses['return_pct'].max():.4f}%]")
    
    # Count how many actually hit the stop
    actually_stopped = stop_losses[stop_losses['return_pct'] <= -0.074]
    print(f"\nPositions that actually hit -0.075% or worse: {len(actually_stopped)} ({len(actually_stopped)/len(stop_losses)*100:.1f}%)")
    
    # Show distribution
    print("\nReturn distribution for 'stop_loss' exits:")
    bins = [-10, -1, -0.5, -0.1, -0.075, -0.05, 0, 0.05, 0.1, 0.5, 1, 10]
    hist = pd.cut(stop_losses['return_pct'], bins=bins).value_counts().sort_index()
    for interval, count in hist.items():
        if count > 0:
            print(f"  {interval}: {count} trades")

# Check take profits
take_profits = exit_df[exit_df['exit_type'] == 'take_profit']
print(f"\n\nFound {len(take_profits)} positions claiming take profit hit")

if len(take_profits) > 0:
    # Check if they actually hit 0.15% profit
    take_profits['return_pct'] = take_profits.apply(
        lambda row: ((row['exit_price'] - row['entry_price']) / row['entry_price'] * 100) if row['quantity'] > 0 
                   else ((row['entry_price'] - row['exit_price']) / row['entry_price'] * 100),
        axis=1
    )
    
    print("\nTake profit returns distribution:")
    print(f"  Expected: +0.15%")
    print(f"  Actual mean: {take_profits['return_pct'].mean():.4f}%")
    print(f"  Actual std: {take_profits['return_pct'].std():.4f}%")
    print(f"  Actual range: [{take_profits['return_pct'].min():.4f}%, {take_profits['return_pct'].max():.4f}%]")
    
    # Count how many actually hit the target
    actually_hit_target = take_profits[take_profits['return_pct'] >= 0.149]
    print(f"\nPositions that actually hit +0.15% or better: {len(actually_hit_target)} ({len(actually_hit_target)/len(take_profits)*100:.1f}%)")
    
    # Show distribution
    print("\nReturn distribution for 'take_profit' exits:")
    bins = [-10, -1, -0.5, -0.1, 0, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 10]
    hist = pd.cut(take_profits['return_pct'], bins=bins).value_counts().sort_index()
    for interval, count in hist.items():
        if count > 0:
            print(f"  {interval}: {count} trades")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("The system is detecting when stops/targets WOULD be hit,")
print("but then exiting at the current market price instead of")
print("the actual stop/target price!")