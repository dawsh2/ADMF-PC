#!/usr/bin/env python3
"""
Fix returns in your existing trades_df
"""
import pandas as pd
import json

def fix_trades_df_returns(trades_df, metadata_df):
    """
    Fix SHORT position returns in existing trades_df
    
    Args:
        trades_df: Your existing trades DataFrame
        metadata_df: The df with metadata containing position direction (quantity sign)
    """
    
    print("=== Fixing Returns in trades_df ===\n")
    
    # Create a copy to avoid modifying original
    fixed_trades = trades_df.copy()
    
    # Track fixes
    fixes_made = 0
    
    # For each trade, find its direction from metadata
    for idx in fixed_trades.index:
        trade = fixed_trades.loc[idx]
        
        # Find matching metadata entry
        matches = metadata_df[
            (abs(metadata_df['entry_price'] - trade['entry_price']) < 0.01) &
            (abs(metadata_df['exit_price'] - trade['exit_price']) < 0.01)
        ]
        
        if len(matches) > 0:
            # Get metadata
            metadata = matches.iloc[0]['metadata']
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            # Get signed quantity to determine direction
            quantity = metadata.get('quantity', 0)
            
            if quantity < 0:  # SHORT position
                # Recalculate return with correct formula
                correct_return = (trade['entry_price'] - trade['exit_price']) / trade['entry_price'] * 100
                fixed_trades.loc[idx, 'return_pct'] = correct_return
                fixed_trades.loc[idx, 'direction'] = 'SHORT'
                fixes_made += 1
            else:
                fixed_trades.loc[idx, 'direction'] = 'LONG'
                # LONG returns are already correct
    
    print(f"Fixed {fixes_made} SHORT position returns\n")
    
    # Recalculate derived columns
    fixed_trades['return_per_bar'] = fixed_trades['return_pct'] / fixed_trades['bars_held']
    
    # Show before/after comparison
    print("=== Before/After Comparison ===")
    print(f"Original mean return: {trades_df['return_pct'].mean():.4f}%")
    print(f"Fixed mean return: {fixed_trades['return_pct'].mean():.4f}%")
    print()
    
    # By direction
    for direction in ['LONG', 'SHORT']:
        dir_trades = fixed_trades[fixed_trades['direction'] == direction]
        if len(dir_trades) > 0:
            print(f"{direction} positions ({len(dir_trades)} trades):")
            print(f"  Mean return: {dir_trades['return_pct'].mean():.4f}%")
            print(f"  Win rate: {(dir_trades['return_pct'] > 0).mean()*100:.1f}%")
    
    # Performance summary
    print("\n=== Performance Summary ===")
    returns_decimal = fixed_trades['return_pct'] / 100
    compounded = (1 + returns_decimal).prod() - 1
    win_rate = (fixed_trades['return_pct'] > 0).mean() * 100
    
    print(f"Total compounded return: {compounded*100:.2f}%")
    print(f"Overall win rate: {win_rate:.1f}%")
    print(f"Average return per trade: {fixed_trades['return_pct'].mean():.4f}%")
    
    # By exit type
    print("\n=== By Exit Type ===")
    for exit_type in fixed_trades['exit_type'].unique():
        type_trades = fixed_trades[fixed_trades['exit_type'] == exit_type]
        print(f"{exit_type}: {len(type_trades)} trades, "
              f"mean {type_trades['return_pct'].mean():.4f}%, "
              f"win rate {(type_trades['return_pct'] > 0).mean()*100:.1f}%")
    
    return fixed_trades

# Usage
print("To fix your trades_df:")
print("fixed_trades = fix_trades_df_returns(trades_df, df)")
print("\nThis will correct the SHORT position returns and show you the true performance.")