#!/usr/bin/env python3
"""
Analyze signal directions and their outcomes
"""
import pandas as pd
import numpy as np

def analyze_signal_directions(df, trades_df):
    """
    Analyze trades based on signal directions
    df: signals dataframe with 'val' column containing signal values
    trades_df: trades dataframe
    """
    
    print("=== Signal Direction Analysis ===\n")
    
    # First, let's see what signals we have
    print("Signal values in data:")
    print(df['val'].value_counts().sort_index())
    print()
    
    # Get entry signals (non-zero values)
    entry_signals = df[df['val'] != 0].copy()
    print(f"Total entry signals: {len(entry_signals)}")
    print(f"Long signals (val > 0): {len(entry_signals[entry_signals['val'] > 0])}")
    print(f"Short signals (val < 0): {len(entry_signals[entry_signals['val'] < 0])}")
    print()
    
    # Now let's match trades to their signal direction
    # We need to find the signal that triggered each trade
    trades_with_direction = []
    
    for idx, trade in trades_df.iterrows():
        # Find the signal closest to entry time
        entry_bar = trade['entry_bar']
        
        # Get signal at entry bar
        signal_at_entry = df[df['idx'] == entry_bar]
        if len(signal_at_entry) > 0:
            signal_val = signal_at_entry.iloc[0]['val']
            direction = 'LONG' if signal_val > 0 else 'SHORT' if signal_val < 0 else 'FLAT'
            
            trade_with_dir = trade.to_dict()
            trade_with_dir['signal_direction'] = direction
            trade_with_dir['signal_value'] = signal_val
            trades_with_direction.append(trade_with_dir)
    
    trades_dir_df = pd.DataFrame(trades_with_direction)
    
    # Analyze by direction
    print("\n=== Returns by Signal Direction ===")
    for direction in ['LONG', 'SHORT']:
        dir_trades = trades_dir_df[trades_dir_df['signal_direction'] == direction]
        if len(dir_trades) > 0:
            print(f"\n{direction} trades ({len(dir_trades)} total):")
            
            # Check return distribution
            print(f"  Mean return: {dir_trades['return_pct'].mean():.4%}")
            print(f"  Returns at -0.1%: {len(dir_trades[abs(dir_trades['return_pct'] + 0.001) < 0.0001])}")
            print(f"  Returns at -0.075%: {len(dir_trades[abs(dir_trades['return_pct'] + 0.00075) < 0.0001])}")
            print(f"  Returns at +0.075%: {len(dir_trades[abs(dir_trades['return_pct'] - 0.00075) < 0.0001])}")
            print(f"  Returns at +0.1%: {len(dir_trades[abs(dir_trades['return_pct'] - 0.001) < 0.0001])}")
            
            # Check exit types
            print(f"\n  Exit types:")
            print(f"    {dir_trades['exit_type'].value_counts().to_dict()}")
            
            # For trades hitting specific levels, show details
            tp_at_loss = dir_trades[
                (abs(dir_trades['return_pct'] + 0.001) < 0.0001) & 
                (dir_trades['exit_type'] == 'take_profit')
            ]
            if len(tp_at_loss) > 0:
                print(f"\n  ⚠️  {len(tp_at_loss)} {direction} trades hit 'take_profit' at -0.1% loss!")
                
            sl_at_gain = dir_trades[
                (abs(dir_trades['return_pct'] - 0.00075) < 0.0001) & 
                (dir_trades['exit_type'] == 'stop_loss')
            ]
            if len(sl_at_gain) > 0:
                print(f"  ⚠️  {len(sl_at_gain)} {direction} trades hit 'stop_loss' at +0.075% gain!")
    
    print("\n=== DIAGNOSIS ===")
    
    # Check which direction has inverted logic
    long_trades = trades_dir_df[trades_dir_df['signal_direction'] == 'LONG']
    short_trades = trades_dir_df[trades_dir_df['signal_direction'] == 'SHORT']
    
    # Count problematic exits for each direction
    long_tp_losses = len(long_trades[
        (abs(long_trades['return_pct'] + 0.001) < 0.0001) & 
        (long_trades['exit_type'] == 'take_profit')
    ])
    short_tp_losses = len(short_trades[
        (abs(short_trades['return_pct'] + 0.001) < 0.0001) & 
        (short_trades['exit_type'] == 'take_profit')
    ])
    
    if long_tp_losses > short_tp_losses:
        print("❌ LONG positions have inverted stop/target logic")
        print("   They hit 'take_profit' when losing money")
    elif short_tp_losses > long_tp_losses:
        print("❌ SHORT positions have inverted stop/target logic")
        print("   They hit 'take_profit' when losing money")
    
    return trades_dir_df

# Usage example:
print("Run this analysis with:")
print("trades_with_dir = analyze_signal_directions(df, trades_df)")
print()
print("This will show which direction (LONG or SHORT) has the inverted logic")