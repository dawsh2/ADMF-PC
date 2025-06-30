#!/usr/bin/env python3
"""Compare position/trade results between notebook and latest results"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load position data from both runs
def load_positions(base_path):
    """Load position open and close events"""
    traces_dir = base_path / 'traces'
    
    # Position events
    pos_open_path = traces_dir / 'portfolio' / 'positions_open' / 'positions_open.parquet'
    pos_close_path = traces_dir / 'portfolio' / 'positions_close' / 'positions_close.parquet'
    
    positions = {}
    if pos_open_path.exists():
        positions['open'] = pd.read_parquet(pos_open_path)
        print(f"  Loaded {len(positions['open'])} position opens")
    
    if pos_close_path.exists():
        positions['close'] = pd.read_parquet(pos_close_path)
        print(f"  Loaded {len(positions['close'])} position closes")
        
    return positions

# Load current run (notebook)
print("=== CURRENT RUN (NOTEBOOK) ===")
current_positions = load_positions(Path('.'))

# Load latest results
print("\n=== LATEST SYSTEM RESULTS ===")
latest_positions = load_positions(Path('../latest'))

# Compare position counts
print("\n=== POSITION COUNT COMPARISON ===")
if 'open' in current_positions and 'open' in latest_positions:
    print(f"Current opens: {len(current_positions['open'])}")
    print(f"Latest opens: {len(latest_positions['open'])}")
    print(f"Difference: {len(latest_positions['open']) - len(current_positions['open'])}")

if 'close' in current_positions and 'close' in latest_positions:
    print(f"\nCurrent closes: {len(current_positions['close'])}")
    print(f"Latest closes: {len(latest_positions['close'])}")
    print(f"Difference: {len(latest_positions['close']) - len(current_positions['close'])}")

# Analyze exit types
print("\n=== EXIT TYPE COMPARISON ===")
if 'close' in current_positions and 'close' in latest_positions:
    curr_close = current_positions['close']
    late_close = latest_positions['close']
    
    if 'exit_type' in curr_close.columns:
        print("\nCurrent run exit types:")
        curr_exits = curr_close['exit_type'].value_counts()
        for exit_type, count in curr_exits.items():
            pct = count/len(curr_close)*100
            print(f"  {exit_type}: {count} ({pct:.1f}%)")
            
    if 'exit_type' in late_close.columns:
        print("\nLatest run exit types:")
        late_exits = late_close['exit_type'].value_counts()
        for exit_type, count in late_exits.items():
            pct = count/len(late_close)*100
            print(f"  {exit_type}: {count} ({pct:.1f}%)")

# Calculate returns for both
print("\n=== RETURN COMPARISON ===")

def calculate_returns(positions):
    """Calculate returns from position events"""
    if 'open' not in positions or 'close' not in positions:
        return None
        
    opens = positions['open']
    closes = positions['close']
    
    # Simple sequential matching
    min_len = min(len(opens), len(closes))
    if min_len == 0:
        return None
        
    # Extract prices
    entry_prices = opens['entry_price'].iloc[:min_len].values if 'entry_price' in opens.columns else opens['px'].iloc[:min_len].values
    exit_prices = closes['exit_price'].iloc[:min_len].values if 'exit_price' in closes.columns else closes['px'].iloc[:min_len].values
    
    # Calculate returns (assuming long positions)
    returns = (exit_prices - entry_prices) / entry_prices * 100
    
    return returns

current_returns = calculate_returns(current_positions)
latest_returns = calculate_returns(latest_positions)

if current_returns is not None:
    print(f"\nCurrent run returns:")
    print(f"  Total trades: {len(current_returns)}")
    print(f"  Mean return: {current_returns.mean():.3f}%")
    print(f"  Std return: {current_returns.std():.3f}%")
    print(f"  Total return: {current_returns.sum():.2f}%")
    print(f"  Win rate: {(current_returns > 0).sum() / len(current_returns) * 100:.1f}%")
    print(f"  Max win: {current_returns.max():.3f}%")
    print(f"  Max loss: {current_returns.min():.3f}%")

if latest_returns is not None:
    print(f"\nLatest run returns:")
    print(f"  Total trades: {len(latest_returns)}")
    print(f"  Mean return: {latest_returns.mean():.3f}%")
    print(f"  Std return: {latest_returns.std():.3f}%")
    print(f"  Total return: {latest_returns.sum():.2f}%")
    print(f"  Win rate: {(latest_returns > 0).sum() / len(latest_returns) * 100:.1f}%")
    print(f"  Max win: {latest_returns.max():.3f}%")
    print(f"  Max loss: {latest_returns.min():.3f}%")

# Check for specific differences in the first few trades
print("\n=== FIRST 10 TRADES COMPARISON ===")
if 'close' in current_positions and 'close' in latest_positions:
    curr_close = current_positions['close']
    late_close = latest_positions['close']
    
    print("\nCurrent run first 10 closes:")
    # Build column list based on what's available
    cols = ['idx']
    if 'exit_type' in curr_close.columns:
        cols.append('exit_type')
    if 'exit_price' in curr_close.columns:
        cols.append('exit_price')
    elif 'px' in curr_close.columns:
        cols.append('px')
    if 'realized_pnl' in curr_close.columns:
        cols.append('realized_pnl')
    print(curr_close[cols].head(10))
    
    print("\nLatest run first 10 closes:")
    # Build column list based on what's available
    cols = ['idx']
    if 'exit_type' in late_close.columns:
        cols.append('exit_type')
    if 'exit_price' in late_close.columns:
        cols.append('exit_price')
    elif 'px' in late_close.columns:
        cols.append('px')
    if 'realized_pnl' in late_close.columns:
        cols.append('realized_pnl')
    print(late_close[cols].head(10))

# Check timing of exits
print("\n=== EXIT TIMING ANALYSIS ===")
if 'open' in current_positions and 'close' in current_positions:
    opens = current_positions['open']
    closes = current_positions['close']
    
    # Calculate hold times
    min_len = min(len(opens), len(closes))
    if min_len > 0:
        entry_bars = opens['idx'].iloc[:min_len].values
        exit_bars = closes['idx'].iloc[:min_len].values
        hold_times = exit_bars - entry_bars
        
        print(f"\nCurrent run hold times:")
        print(f"  Mean: {hold_times.mean():.1f} bars")
        print(f"  Median: {np.median(hold_times):.0f} bars")
        print(f"  Min: {hold_times.min()} bars")
        print(f"  Max: {hold_times.max()} bars")
        
        # Count immediate exits
        immediate_exits = (hold_times == 1).sum()
        print(f"  Immediate exits (1 bar): {immediate_exits} ({immediate_exits/len(hold_times)*100:.1f}%)")