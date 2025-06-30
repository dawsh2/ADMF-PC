#!/usr/bin/env python3
"""Find which trades are missing between runs"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load position data
print("Loading position data...")

# Current run (notebook)
curr_opens = pd.read_parquet('traces/portfolio/positions_open/positions_open.parquet')
curr_closes = pd.read_parquet('traces/portfolio/positions_close/positions_close.parquet')

# Latest run
late_opens = pd.read_parquet('../latest/traces/portfolio/positions_open/positions_open.parquet')
late_closes = pd.read_parquet('../latest/traces/portfolio/positions_close/positions_close.parquet')

print(f"Current run: {len(curr_opens)} positions")
print(f"Latest run: {len(late_opens)} positions")
print(f"Difference: {len(curr_opens) - len(late_opens)} positions")

# Find missing positions by comparing indices
curr_open_idx = set(curr_opens['idx'].values)
late_open_idx = set(late_opens['idx'].values)

missing_in_latest = curr_open_idx - late_open_idx
extra_in_latest = late_open_idx - curr_open_idx

print(f"\n=== MISSING POSITIONS ===")
print(f"Missing in latest run: {len(missing_in_latest)} positions")
print(f"Extra in latest run: {len(extra_in_latest)} positions")

if missing_in_latest:
    print("\nPositions in current but NOT in latest:")
    missing_df = curr_opens[curr_opens['idx'].isin(missing_in_latest)].sort_values('idx')
    
    # Build column list based on what's available
    cols = ['idx']
    if 'entry_price' in missing_df.columns:
        cols.append('entry_price')
    elif 'px' in missing_df.columns:
        cols.append('px')
    if 'quantity' in missing_df.columns:
        cols.append('quantity')
    if 'strategy_id' in missing_df.columns:
        cols.append('strategy_id')
    
    print(missing_df[cols].head(20))
    
    # Check what happened at those bars
    print("\n=== INVESTIGATING MISSING POSITIONS ===")
    
    # Load signals to check
    curr_signals = pd.read_parquet('traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')
    late_signals = pd.read_parquet('../latest/traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet')
    
    for idx in list(missing_in_latest)[:5]:  # Check first 5
        print(f"\nPosition at bar {idx}:")
        
        # Check signals around this position
        signal_window = curr_signals[(curr_signals['idx'] >= idx-2) & (curr_signals['idx'] <= idx+2)]
        print("Current run signals:")
        print(signal_window[['idx', 'val', 'px']])
        
        # Check if this position was closed quickly
        closed = curr_closes[curr_closes['idx'] == idx+1]  # Check immediate close
        if not closed.empty:
            print(f"Position closed immediately at bar {idx+1}:")
            print(closed[['idx', 'exit_type', 'exit_price']].iloc[0])

# Check for patterns in missing trades
print("\n=== TIMING ANALYSIS ===")
if missing_in_latest:
    missing_bars = sorted(list(missing_in_latest))
    print(f"Missing position bars: {missing_bars[:20]}...")
    
    # Check if they cluster together
    gaps = np.diff(missing_bars)
    print(f"\nGaps between missing positions:")
    print(f"  Mean gap: {gaps.mean():.1f} bars")
    print(f"  Median gap: {np.median(gaps):.0f} bars")
    print(f"  Min gap: {gaps.min()} bars")
    print(f"  Max gap: {gaps.max()} bars")

# Compare total performance impact
print("\n=== PERFORMANCE IMPACT ===")

# Calculate returns for missing trades
if missing_in_latest:
    missing_opens = curr_opens[curr_opens['idx'].isin(missing_in_latest)]
    
    # Find corresponding closes
    total_pnl = 0
    found_closes = 0
    
    for _, open_pos in missing_opens.iterrows():
        # Find the close for this position
        # Assuming positions are closed in order
        open_idx = open_pos['idx']
        close_candidates = curr_closes[curr_closes['idx'] > open_idx]
        
        if not close_candidates.empty:
            close_pos = close_candidates.iloc[0]
            
            entry_price = open_pos['entry_price'] if 'entry_price' in open_pos else open_pos['px']
            exit_price = close_pos['exit_price'] if 'exit_price' in close_pos else close_pos['px']
            
            pnl = (exit_price - entry_price) / entry_price * 100
            total_pnl += pnl
            found_closes += 1
            
            if found_closes <= 5:  # Show first 5
                print(f"Missing trade: entry={entry_price:.2f}, exit={exit_price:.2f}, pnl={pnl:.3f}%")
    
    if found_closes > 0:
        print(f"\nTotal PnL from {found_closes} missing trades: {total_pnl:.2f}%")
        print(f"Average PnL per missing trade: {total_pnl/found_closes:.3f}%")