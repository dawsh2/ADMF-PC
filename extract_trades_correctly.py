#!/usr/bin/env python3
"""
Extract trades from trace files with CORRECT return calculations for both LONG and SHORT positions
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def extract_trades_from_traces(workspace_path):
    """Extract trades with correct return calculations"""
    
    print("=== Extracting Trades with Correct Returns ===\n")
    
    # Find trace files
    pattern = str(Path(workspace_path) / "traces" / "*" / "trace_*.parquet")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No trace files found in {workspace_path}")
        return None
    
    print(f"Found {len(files)} trace files")
    
    # Load and combine traces
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    
    traces = pd.concat(dfs, ignore_index=True)
    
    # Get position events
    entries = traces[traces['event_type'] == 'POSITION_OPENED'].copy()
    exits = traces[traces['event_type'] == 'POSITION_CLOSED'].copy()
    
    print(f"Found {len(entries)} position entries and {len(exits)} position exits")
    
    # Extract trades
    trades = []
    
    for _, exit in exits.iterrows():
        # Find matching entry
        strategy_id = exit['strategy_id']
        exit_time = exit['timestamp']
        
        # Get the most recent entry before this exit
        entry_mask = (
            (entries['strategy_id'] == strategy_id) &
            (entries['timestamp'] < exit_time)
        )
        
        if entry_mask.any():
            # Get the most recent entry
            matching_entries = entries[entry_mask].sort_values('timestamp')
            entry = matching_entries.iloc[-1]
            
            # Extract metadata
            entry_meta = entry.get('metadata', {})
            exit_meta = exit.get('metadata', {})
            
            if isinstance(entry_meta, str):
                import json
                entry_meta = json.loads(entry_meta)
            if isinstance(exit_meta, str):
                import json
                exit_meta = json.loads(exit_meta)
            
            # Get prices
            entry_price = float(entry.get('price', 0))
            exit_price = float(exit.get('price', 0))
            
            # Get quantity/direction
            quantity = entry_meta.get('quantity', 1)
            
            # Calculate return with CORRECT formula
            if quantity > 0:  # LONG position
                return_pct = (exit_price - entry_price) / entry_price * 100
            else:  # SHORT position
                return_pct = (entry_price - exit_price) / entry_price * 100
            
            # Get bar indices
            entry_bar = entry.get('bar_index', 0)
            exit_bar = exit.get('bar_index', 0)
            bars_held = exit_bar - entry_bar
            
            # Build trade record
            trade = {
                'entry_bar': entry_bar,
                'exit_bar': exit_bar,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': abs(quantity) * 100,  # Convert to match your format
                'signed_quantity': quantity,  # Keep signed version
                'direction': 'LONG' if quantity > 0 else 'SHORT',
                'realized_pnl': (exit_price - entry_price) * abs(quantity) * 100,
                'exit_type': exit_meta.get('exit_type', 'unknown'),
                'strategy_id': strategy_id,
                'bars_held': bars_held,
                'return_pct': return_pct,
                'return_per_bar': return_pct / bars_held if bars_held > 0 else return_pct,
                'entry_time': entry['timestamp'],
                'exit_time': exit['timestamp']
            }
            
            trades.append(trade)
    
    # Create DataFrame
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) > 0:
        # Add return bucket
        trades_df['return_bucket'] = pd.cut(
            trades_df['return_pct'], 
            bins=[-np.inf, -0.1, -0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.1, np.inf],
            labels=['-0.1+', '-0.1', '-0.075', '-0.05', '-0.025', '0', '0.025', '0.05', '0.075', '0.1+']
        )
        
        # Sort by entry time
        trades_df = trades_df.sort_values('entry_time').reset_index(drop=True)
    
    return trades_df

def analyze_corrected_trades(trades_df):
    """Analyze trades with correct returns"""
    
    print("\n=== Analysis with Correct Returns ===\n")
    
    # Overall stats
    print(f"Total trades: {len(trades_df)}")
    print(f"Mean return per trade: {trades_df['return_pct'].mean():.4f}%")
    print()
    
    # By direction
    for direction in ['LONG', 'SHORT']:
        dir_trades = trades_df[trades_df['direction'] == direction]
        if len(dir_trades) > 0:
            print(f"{direction} positions ({len(dir_trades)} trades):")
            print(f"  Mean return: {dir_trades['return_pct'].mean():.4f}%")
            print(f"  Win rate: {(dir_trades['return_pct'] > 0).mean()*100:.1f}%")
            
            # By exit type
            for exit_type in dir_trades['exit_type'].unique():
                type_trades = dir_trades[dir_trades['exit_type'] == exit_type]
                print(f"  {exit_type}: {len(type_trades)} trades, "
                      f"mean return {type_trades['return_pct'].mean():.4f}%")
            print()
    
    # Calculate cumulative performance
    print("=== Cumulative Performance ===")
    returns_decimal = trades_df['return_pct'] / 100
    compounded = (1 + returns_decimal).prod() - 1
    print(f"Total compounded return: {compounded*100:.2f}%")
    print(f"Overall win rate: {(trades_df['return_pct'] > 0).mean()*100:.1f}%")
    
    # Compare to notebook
    print("\nYour notebook showed: 10.27% returns, 75% win rate")
    print(f"Corrected backtest shows: {compounded*100:.2f}% returns, "
          f"{(trades_df['return_pct'] > 0).mean()*100:.1f}% win rate")
    
    return trades_df

# Usage
if __name__ == "__main__":
    print("To use this:")
    print("1. corrected_trades = extract_trades_from_traces('path/to/workspace')")
    print("2. analyze_corrected_trades(corrected_trades)")
    print("\nOr in your notebook:")
    print("%run /Users/daws/ADMF-PC/extract_trades_correctly.py")
    print("corrected_trades = extract_trades_from_traces('results/latest')")
    print("analyze_corrected_trades(corrected_trades)")