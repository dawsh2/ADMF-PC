#!/usr/bin/env python3
"""
Check signal data structure and match with trades
"""
import pandas as pd
import numpy as np

# First, let's see what data we have
print("=== Checking DataFrame Structures ===\n")

print("1. Main df columns:")
print("   Run: print(df.columns.tolist())")
print("   Run: print(df.head())")
print()

print("2. Trades df columns:")  
print("   Run: print(trades_df.columns.tolist())")
print("   Run: print(trades_df.head())")
print()

print("3. If you have market_data:")
print("   Run: print(market_data.columns.tolist())")
print()

# Function to analyze once we know the structure
def analyze_with_signals(signals_df, trades_df, signal_col='signal'):
    """
    Analyze trades based on signal directions
    signals_df: DataFrame with signals (1 for long, -1 for short)
    trades_df: trades DataFrame
    signal_col: name of the signal column
    """
    
    print("=== Signal Direction Analysis ===\n")
    
    # Check signal distribution
    print(f"Signal values in {signal_col} column:")
    print(signals_df[signal_col].value_counts().sort_index())
    print()
    
    # For each trade, find its signal direction
    trades_with_direction = []
    
    for idx, trade in trades_df.iterrows():
        # Get the entry bar/time
        entry_bar = trade.get('entry_bar', None)
        entry_time = trade.get('entry_time', None)
        
        # Find corresponding signal
        if entry_bar is not None:
            # Match by bar index
            signal_rows = signals_df[signals_df.index == entry_bar]
        elif entry_time is not None:
            # Match by timestamp
            signal_rows = signals_df[signals_df['timestamp'] == entry_time]
        else:
            continue
            
        if len(signal_rows) > 0:
            signal_val = signal_rows.iloc[0][signal_col]
            direction = 'LONG' if signal_val > 0 else 'SHORT' if signal_val < 0 else 'FLAT'
            
            trade_dict = trade.to_dict()
            trade_dict['signal_direction'] = direction
            trade_dict['signal_value'] = signal_val
            trades_with_direction.append(trade_dict)
    
    trades_dir_df = pd.DataFrame(trades_with_direction)
    
    # Analyze by direction
    print("\n=== Returns by Signal Direction ===")
    for direction in ['LONG', 'SHORT']:
        dir_trades = trades_dir_df[trades_dir_df['signal_direction'] == direction]
        if len(dir_trades) > 0:
            print(f"\n{direction} trades ({len(dir_trades)} total):")
            
            # Check return distribution
            print(f"  Mean return: {dir_trades['return_pct'].mean():.4%}")
            
            # Count trades at specific levels
            at_neg_100 = len(dir_trades[abs(dir_trades['return_pct'] + 0.001) < 0.0001])
            at_neg_75 = len(dir_trades[abs(dir_trades['return_pct'] + 0.00075) < 0.0001])
            at_pos_75 = len(dir_trades[abs(dir_trades['return_pct'] - 0.00075) < 0.0001])
            at_pos_100 = len(dir_trades[abs(dir_trades['return_pct'] - 0.001) < 0.0001])
            
            print(f"  Returns at -0.1%: {at_neg_100}")
            print(f"  Returns at -0.075%: {at_neg_75}")
            print(f"  Returns at +0.075%: {at_pos_75}")
            print(f"  Returns at +0.1%: {at_pos_100}")
            
            # Check exit types
            if 'exit_type' in dir_trades.columns:
                print(f"\n  Exit types:")
                for exit_type, count in dir_trades['exit_type'].value_counts().items():
                    print(f"    {exit_type}: {count}")
                
                # Check for inverted logic
                tp_losses = dir_trades[
                    (abs(dir_trades['return_pct'] + 0.001) < 0.0001) & 
                    (dir_trades['exit_type'] == 'take_profit')
                ]
                if len(tp_losses) > 0:
                    print(f"\n  ⚠️  {len(tp_losses)} {direction} trades hit 'take_profit' at -0.1% LOSS!")
                    print(f"     This means {direction} stop/target logic is INVERTED")
    
    return trades_dir_df

print("\n=== How to proceed ===")
print("1. First show me your DataFrame structures")
print("2. Then we'll match signals to trades")
print("3. This will reveal which direction has inverted logic")