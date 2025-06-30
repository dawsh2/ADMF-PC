#!/usr/bin/env python3
"""
Calculate the CORRECT returns accounting for SHORT position direction
"""
import pandas as pd
import json
import numpy as np

def calculate_correct_returns(df, trades_df):
    """Recalculate returns with correct formula for SHORT positions"""
    
    print("=== Recalculating Returns with Correct SHORT Formula ===\n")
    
    # Create a copy to avoid modifying original
    corrected_trades = trades_df.copy()
    
    # Track corrections
    corrections_made = 0
    
    # Fix each trade
    for idx in corrected_trades.index:
        trade = corrected_trades.loc[idx]
        
        # Find direction from metadata
        matches = df[
            (abs(df['entry_price'] - trade['entry_price']) < 0.01) &
            (abs(df['exit_price'] - trade['exit_price']) < 0.01)
        ]
        
        if len(matches) > 0:
            metadata = matches.iloc[0]['metadata']
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            quantity = metadata.get('quantity', 0)
            
            if quantity < 0:  # SHORT position
                # Recalculate with correct formula
                correct_return = (trade['entry_price'] - trade['exit_price']) / trade['entry_price']
                corrected_trades.loc[idx, 'return_pct'] = correct_return
                corrected_trades.loc[idx, 'direction'] = 'SHORT'
                corrections_made += 1
            else:
                corrected_trades.loc[idx, 'direction'] = 'LONG'
    
    print(f"Corrected {corrections_made} SHORT position returns\n")
    
    # Compare results
    print("=== Original vs Corrected Returns ===")
    print(f"Original mean return: {trades_df['return_pct'].mean():.4%}")
    print(f"Corrected mean return: {corrected_trades['return_pct'].mean():.4%}")
    print()
    
    # By direction
    for direction in ['LONG', 'SHORT']:
        dir_trades = corrected_trades[corrected_trades['direction'] == direction]
        if len(dir_trades) > 0:
            print(f"\n{direction} positions ({len(dir_trades)} trades):")
            print(f"  Mean return: {dir_trades['return_pct'].mean():.4%}")
            print(f"  Win rate: {(dir_trades['return_pct'] > 0).mean():.1%}")
            print(f"  Total return: {dir_trades['return_pct'].sum():.2%}")
    
    # Calculate cumulative returns
    print("\n=== Cumulative Performance ===")
    
    # Simple cumulative (sum of returns)
    total_return_simple = corrected_trades['return_pct'].sum()
    print(f"Simple cumulative return: {total_return_simple:.2%}")
    
    # Compounded returns
    cumulative_return = (1 + corrected_trades['return_pct']).prod() - 1
    print(f"Compounded cumulative return: {cumulative_return:.2%}")
    
    # Win/loss statistics
    print("\n=== Win/Loss Statistics ===")
    wins = corrected_trades[corrected_trades['return_pct'] > 0]
    losses = corrected_trades[corrected_trades['return_pct'] < 0]
    
    print(f"Total trades: {len(corrected_trades)}")
    print(f"Winners: {len(wins)} ({len(wins)/len(corrected_trades)*100:.1f}%)")
    print(f"Losers: {len(losses)} ({len(losses)/len(corrected_trades)*100:.1f}%)")
    print(f"Average win: {wins['return_pct'].mean():.4%}")
    print(f"Average loss: {losses['return_pct'].mean():.4%}")
    
    # By exit type
    print("\n=== Performance by Exit Type ===")
    for exit_type in corrected_trades['exit_type'].unique():
        type_trades = corrected_trades[corrected_trades['exit_type'] == exit_type]
        print(f"\n{exit_type} ({len(type_trades)} trades):")
        print(f"  Mean return: {type_trades['return_pct'].mean():.4%}")
        print(f"  Win rate: {(type_trades['return_pct'] > 0).mean():.1%}")
    
    # Compare to notebook
    print("\n=== Comparison to Notebook ===")
    print("Your notebook showed:")
    print("  - 10.27% returns")
    print("  - 75% win rate")
    print("\nCorrected backtest shows:")
    print(f"  - {cumulative_return:.2%} returns")
    print(f"  - {len(wins)/len(corrected_trades)*100:.1f}% win rate")
    
    return corrected_trades

# Run the correction
corrected_trades = calculate_correct_returns(df, trades_df)