#!/usr/bin/env python3
"""Analyze performance with correct return calculation for short positions."""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("=== Performance Analysis with Fixed Short Position Returns ===")

# Load position events
results_dir = Path("config/bollinger/results/latest")
pos_open_file = results_dir / "traces/portfolio/positions_open/positions_open.parquet"
pos_close_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"

if pos_open_file.exists() and pos_close_file.exists():
    opens = pd.read_parquet(pos_open_file)
    closes = pd.read_parquet(pos_close_file)
    
    # Parse metadata to get quantity
    for df in [opens, closes]:
        if 'metadata' in df.columns:
            for i in range(len(df)):
                if isinstance(df.iloc[i]['metadata'], str):
                    try:
                        meta = json.loads(df.iloc[i]['metadata'])
                        for key, value in meta.items():
                            if key not in df.columns:
                                df.loc[df.index[i], key] = value
                    except:
                        pass
    
    # Match trades
    min_len = min(len(opens), len(closes))
    if min_len > 0:
        trades_df = pd.DataFrame({
            'entry_bar': opens['idx'].iloc[:min_len].values,
            'exit_bar': closes['idx'].iloc[:min_len].values,
            'entry_price': opens['entry_price'].iloc[:min_len].values if 'entry_price' in opens.columns else opens['px'].iloc[:min_len].values,
            'exit_price': closes['exit_price'].iloc[:min_len].values if 'exit_price' in closes.columns else closes['px'].iloc[:min_len].values,
            'quantity': opens['quantity'].iloc[:min_len].values if 'quantity' in opens.columns else 100,
            'exit_type': closes['exit_type'].iloc[:min_len].values if 'exit_type' in closes.columns else 'unknown',
            'strategy_id': opens['strategy_id'].iloc[:min_len].values if 'strategy_id' in opens.columns else 'unknown'
        })
        
        # Calculate returns correctly for long/short
        trades_df['direction'] = trades_df['quantity'].apply(lambda x: 'LONG' if x > 0 else 'SHORT')
        
        # Correct return calculation
        trades_df['return_pct'] = trades_df.apply(
            lambda row: ((row['exit_price'] - row['entry_price']) / row['entry_price'] * 100) if row['quantity'] > 0 
                       else ((row['entry_price'] - row['exit_price']) / row['entry_price'] * 100),
            axis=1
        )
        
        # Performance metrics
        print(f"\nTotal trades: {len(trades_df)}")
        print(f"Long trades: {(trades_df['direction'] == 'LONG').sum()}")
        print(f"Short trades: {(trades_df['direction'] == 'SHORT').sum()}")
        
        # Win rate
        winning_trades = (trades_df['return_pct'] > 0).sum()
        win_rate = winning_trades / len(trades_df) * 100
        print(f"\nWin rate: {win_rate:.1f}%")
        
        # Average returns
        avg_return = trades_df['return_pct'].mean()
        print(f"Average return per trade: {avg_return:.3f}%")
        
        # By exit type
        print("\nReturns by exit type:")
        for exit_type in trades_df['exit_type'].unique():
            mask = trades_df['exit_type'] == exit_type
            count = mask.sum()
            avg_ret = trades_df[mask]['return_pct'].mean()
            print(f"  {exit_type}: {count} trades, avg return: {avg_ret:.3f}%")
        
        # Analyze stop losses by direction
        print("\nStop loss analysis by direction:")
        sl_trades = trades_df[trades_df['exit_type'] == 'stop_loss']
        
        for direction in ['LONG', 'SHORT']:
            dir_sl = sl_trades[sl_trades['direction'] == direction]
            if len(dir_sl) > 0:
                print(f"\n{direction} positions with stop loss:")
                print(f"  Count: {len(dir_sl)}")
                print(f"  Average return: {dir_sl['return_pct'].mean():.4f}%")
                print(f"  Expected: -0.075%")
                
                # Show examples
                print(f"  Examples:")
                for i, row in dir_sl.head(3).iterrows():
                    print(f"    Entry: ${row['entry_price']:.4f}, Exit: ${row['exit_price']:.4f}, Return: {row['return_pct']:.4f}%")
        
        # Total return
        total_return = trades_df['return_pct'].sum()
        print(f"\nTotal return (sum of all trades): {total_return:.2f}%")
        
        # Compound return
        compound_return = (1 + trades_df['return_pct'] / 100).prod() - 1
        print(f"Compound return: {compound_return * 100:.2f}%")
        
        # Compare old vs new calculation
        print("\n=== Impact of Fix ===")
        
        # Old calculation (always using exit - entry)
        trades_df['old_return_pct'] = (trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price'] * 100
        
        # Show difference for shorts
        short_sl = trades_df[(trades_df['direction'] == 'SHORT') & (trades_df['exit_type'] == 'stop_loss')]
        if len(short_sl) > 0:
            print(f"\nShort stop losses:")
            print(f"  Old calculation average: {short_sl['old_return_pct'].mean():.4f}%")
            print(f"  New calculation average: {short_sl['return_pct'].mean():.4f}%")
            print(f"  Difference: {(short_sl['return_pct'].mean() - short_sl['old_return_pct'].mean()):.4f}%")
        
        # Overall impact
        old_total = trades_df['old_return_pct'].sum()
        new_total = trades_df['return_pct'].sum()
        print(f"\nTotal return impact:")
        print(f"  Old calculation: {old_total:.2f}%")
        print(f"  New calculation: {new_total:.2f}%")
        print(f"  Difference: {new_total - old_total:.2f}%")
        
else:
    print("Could not find position event files")