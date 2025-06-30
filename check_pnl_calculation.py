#!/usr/bin/env python3
"""
Check why realized PnL is always 0
"""
import pandas as pd

def check_pnl_issues(trades_df):
    """Analyze PnL calculation issues"""
    
    print("=== PnL Calculation Analysis ===\n")
    
    # Show sample trades with their PnL
    print("Sample trades showing the PnL issue:")
    print("-" * 80)
    
    for idx in trades_df.index[:5]:
        trade = trades_df.loc[idx]
        
        # Calculate what PnL should be
        price_diff = trade['exit_price'] - trade['entry_price']
        expected_pnl = price_diff * trade['quantity']
        
        print(f"\nTrade {idx}:")
        print(f"  Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f}")
        print(f"  Price change: ${price_diff:.2f}")
        print(f"  Quantity: {trade['quantity']}")
        print(f"  Return %: {trade['return_pct']:.4%}")
        print(f"  Realized PnL: ${trade['realized_pnl']}")
        print(f"  Expected PnL: ${expected_pnl:.2f}")
        
        if trade['realized_pnl'] == 0:
            print("  ❌ PnL is 0 but should be non-zero!")
    
    print("\n" + "="*60 + "\n")
    
    # Summary statistics
    print("=== Summary ===")
    print(f"Total trades: {len(trades_df)}")
    print(f"Trades with realized_pnl = 0: {len(trades_df[trades_df['realized_pnl'] == 0])}")
    print(f"Trades with non-zero returns: {len(trades_df[trades_df['return_pct'] != 0])}")
    
    print("\n=== Issues Found ===")
    print("1. ❌ Realized PnL is always 0")
    print("2. ❌ Quantity is 100 instead of 1") 
    print("3. ❌ Some trades hit 'take_profit' at -0.1% loss")
    print("4. ❌ Some trades hit 'stop_loss' at +0.075% gain")
    
    print("\n=== Root Cause Analysis ===")
    print("The combination of these issues suggests:")
    print("• PnL calculation is broken (always returns 0)")
    print("• Position sizing is using 100 instead of configured value of 1")
    print("• Stop/Target logic is inverted for one or both directions")
    
    # Check if it's a data type issue
    print(f"\n=== Data Types ===")
    print(f"realized_pnl dtype: {trades_df['realized_pnl'].dtype}")
    print(f"Unique PnL values: {trades_df['realized_pnl'].unique()}")
    
    return trades_df

# Run the check
check_pnl_issues(trades_df)