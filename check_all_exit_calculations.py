#!/usr/bin/env python3
"""Check all exit type calculations for short positions."""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("=== Checking All Exit Type Calculations ===")

# Load position events
results_dir = Path("config/bollinger/results/latest")
pos_open_file = results_dir / "traces/portfolio/positions_open/positions_open.parquet"
pos_close_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"

if pos_open_file.exists() and pos_close_file.exists():
    opens = pd.read_parquet(pos_open_file)
    closes = pd.read_parquet(pos_close_file)
    
    # Parse metadata
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
            'entry_price': opens['entry_price'].iloc[:min_len].values if 'entry_price' in opens.columns else opens['px'].iloc[:min_len].values,
            'exit_price': closes['exit_price'].iloc[:min_len].values if 'exit_price' in closes.columns else closes['px'].iloc[:min_len].values,
            'quantity': opens['quantity'].iloc[:min_len].values if 'quantity' in opens.columns else 100,
            'exit_type': closes['exit_type'].iloc[:min_len].values if 'exit_type' in closes.columns else 'unknown'
        })
        
        # All positions are short in this example (quantity < 0)
        trades_df['direction'] = trades_df['quantity'].apply(lambda x: 'LONG' if x > 0 else 'SHORT')
        
        # Calculate both ways
        trades_df['simple_return'] = (trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price'] * 100
        trades_df['correct_return'] = trades_df.apply(
            lambda row: ((row['exit_price'] - row['entry_price']) / row['entry_price'] * 100) if row['quantity'] > 0 
                       else ((row['entry_price'] - row['exit_price']) / row['entry_price'] * 100),
            axis=1
        )
        
        print(f"\nTotal trades: {len(trades_df)}")
        print(f"All trades are: {trades_df['direction'].iloc[0]}")
        
        # Check each exit type
        for exit_type in ['stop_loss', 'take_profit', 'signal']:
            exit_trades = trades_df[trades_df['exit_type'] == exit_type]
            if len(exit_trades) > 0:
                print(f"\n{exit_type.upper()} ({len(exit_trades)} trades):")
                print(f"  Simple calculation avg: {exit_trades['simple_return'].mean():.4f}%")
                print(f"  Correct calculation avg: {exit_trades['correct_return'].mean():.4f}%")
                
                # For shorts:
                # - Stop loss: price goes UP (bad) = negative return
                # - Take profit: price goes DOWN (good) = positive return  
                # - Signal exit: depends on price movement
                
                if exit_type == 'stop_loss':
                    # For shorts, stop loss means price went up
                    up_moves = (exit_trades['exit_price'] > exit_trades['entry_price']).sum()
                    print(f"  Price went up (expected for short SL): {up_moves}/{len(exit_trades)}")
                    
                elif exit_type == 'take_profit':
                    # For shorts, take profit means price went down
                    down_moves = (exit_trades['exit_price'] < exit_trades['entry_price']).sum()
                    print(f"  Price went down (expected for short TP): {down_moves}/{len(exit_trades)}")
                
                # Show examples
                print(f"  Examples:")
                for i, row in exit_trades.head(3).iterrows():
                    price_move = "UP" if row['exit_price'] > row['entry_price'] else "DOWN"
                    print(f"    Entry: ${row['entry_price']:.2f}, Exit: ${row['exit_price']:.2f} ({price_move})")
                    print(f"      Simple: {row['simple_return']:.4f}%, Correct: {row['correct_return']:.4f}%")
        
        print("\n=== SUMMARY ===")
        print("For SHORT positions:")
        print("- Stop loss (price goes UP): Should show NEGATIVE return")
        print("- Take profit (price goes DOWN): Should show POSITIVE return")
        print("- Signal exit: Depends on price movement")
        print("\nThe 'simple' calculation is WRONG for shorts - it inverts the sign!")
        print("The notebook template needs to be fixed to use the correct calculation.")
        
else:
    print("Could not find position event files")