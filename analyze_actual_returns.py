#!/usr/bin/env python3
"""
Analyze returns understanding they're already in percentage format
"""
import pandas as pd
import json
import numpy as np

def analyze_actual_returns(df, trades_df):
    """Analyze returns knowing they're already percentages"""
    
    print("=== Understanding the Return Format ===\n")
    print("The 'return_pct' values are already in percentage format:")
    print("- 0.1 means 0.1% (not 10%)")
    print("- 0.075 means 0.075% (not 7.5%)")
    print()
    
    # Create corrected trades with proper SHORT calculations
    corrected_trades = trades_df.copy()
    
    # Fix SHORT returns
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
                # Current return is wrong sign, just flip it
                corrected_trades.loc[idx, 'return_pct'] = -trade['return_pct']
                corrected_trades.loc[idx, 'direction'] = 'SHORT'
            else:
                corrected_trades.loc[idx, 'direction'] = 'LONG'
    
    print("=== Corrected Performance Summary ===\n")
    
    # Overall stats
    total_trades = len(corrected_trades)
    mean_return_pct = corrected_trades['return_pct'].mean()
    
    print(f"Total trades: {total_trades}")
    print(f"Mean return per trade: {mean_return_pct:.4f}%")
    print()
    
    # By direction
    for direction in ['LONG', 'SHORT']:
        dir_trades = corrected_trades[corrected_trades['direction'] == direction]
        if len(dir_trades) > 0:
            print(f"{direction} positions ({len(dir_trades)} trades):")
            print(f"  Mean return: {dir_trades['return_pct'].mean():.4f}%")
            print(f"  Median return: {dir_trades['return_pct'].median():.4f}%")
            
            wins = dir_trades[dir_trades['return_pct'] > 0]
            print(f"  Win rate: {len(wins)/len(dir_trades)*100:.1f}%")
            print()
    
    # By exit type with corrected returns
    print("=== Returns by Exit Type (Corrected) ===\n")
    
    exit_stats = corrected_trades.groupby('exit_type').agg({
        'return_pct': ['mean', 'median', 'count', lambda x: (x > 0).mean()]
    }).round(4)
    
    for exit_type in corrected_trades['exit_type'].unique():
        type_trades = corrected_trades[corrected_trades['exit_type'] == exit_type]
        mean_ret = type_trades['return_pct'].mean()
        win_rate = (type_trades['return_pct'] > 0).mean() * 100
        
        print(f"{exit_type}:")
        print(f"  Count: {len(type_trades)}")
        print(f"  Mean return: {mean_ret:.4f}%")
        print(f"  Win rate: {win_rate:.1f}%")
        
        # Check expected values
        if exit_type == 'take_profit':
            print(f"  Expected: ~0.1% (configured take profit)")
        elif exit_type == 'stop_loss':
            print(f"  Expected: ~-0.075% (configured stop loss)")
        print()
    
    # Calculate actual cumulative return
    print("=== Actual Cumulative Performance ===\n")
    
    # Convert percentage to decimal for calculation
    returns_decimal = corrected_trades['return_pct'] / 100
    
    # Simple sum
    simple_return = returns_decimal.sum()
    print(f"Simple sum of returns: {simple_return:.4f} ({simple_return*100:.2f}%)")
    
    # Compounded
    compounded_return = (1 + returns_decimal).prod() - 1
    print(f"Compounded return: {compounded_return:.4f} ({compounded_return*100:.2f}%)")
    
    # Win/loss analysis
    print("\n=== Win/Loss Analysis ===")
    winners = corrected_trades[corrected_trades['return_pct'] > 0]
    losers = corrected_trades[corrected_trades['return_pct'] < 0]
    
    print(f"Winners: {len(winners)} ({len(winners)/total_trades*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/total_trades*100:.1f}%)")
    print(f"Average win: {winners['return_pct'].mean():.4f}%")
    print(f"Average loss: {losers['return_pct'].mean():.4f}%")
    
    # Sharpe ratio approximation
    print("\n=== Risk Metrics ===")
    returns_std = corrected_trades['return_pct'].std()
    sharpe_approx = mean_return_pct / returns_std * np.sqrt(252 * 78)  # 252 days, 78 5-min bars/day
    print(f"Return volatility: {returns_std:.4f}%")
    print(f"Approximate Sharpe ratio: {sharpe_approx:.2f}")
    
    # Issue summary
    print("\n=== Issues Found ===")
    print("1. Position sizing: Using 100 shares instead of 1")
    print("2. PnL calculation: Always returns 0")
    print("3. SHORT returns: Had wrong sign (now corrected)")
    print("\nWith these corrections, the actual performance is:")
    print(f"- {compounded_return*100:.2f}% total return")
    print(f"- {len(winners)/total_trades*100:.1f}% win rate")
    print(f"- vs Notebook: 10.27% return, 75% win rate")
    
    return corrected_trades

# Run analysis
corrected = analyze_actual_returns(df, trades_df)