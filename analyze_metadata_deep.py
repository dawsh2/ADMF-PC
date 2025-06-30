#!/usr/bin/env python3
"""
Deep dive into metadata to find signal direction
"""
import pandas as pd
import json

def analyze_metadata_deep(df, trades_df):
    """Extract signal direction from nested metadata"""
    
    print("=== Deep Metadata Analysis ===\n")
    
    # First, let's look deeper into the metadata structure
    print("Examining metadata structure in detail:")
    
    for i in range(min(5, len(df))):
        metadata = df.iloc[i]['metadata']
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        print(f"\nRow {i} - Exit type: {df.iloc[i]['exit_type']}")
        print(f"Return: {(df.iloc[i]['exit_price'] / df.iloc[i]['entry_price'] - 1) * 100:.3f}%")
        
        # Check nested metadata
        if 'metadata' in metadata:
            nested = metadata['metadata']
            if isinstance(nested, str):
                try:
                    nested = json.loads(nested)
                except:
                    pass
            print(f"Nested metadata type: {type(nested)}")
            if isinstance(nested, dict):
                print(f"Nested keys: {list(nested.keys())}")
                # Look for signal info
                for key in ['signal', 'direction', 'side', 'signal_value', 'position_side']:
                    if key in nested:
                        print(f"  {key}: {nested[key]}")
        
        # Also check top-level metadata
        for key in ['signal', 'direction', 'side', 'quantity']:
            if key in metadata:
                print(f"Top-level {key}: {metadata[key]}")
    
    print("\n" + "="*60 + "\n")
    
    # Now let's analyze by exit type and return
    print("=== Trades by Exit Type and Return ===\n")
    
    # Group trades by exit type
    by_exit_type = trades_df.groupby('exit_type')
    
    for exit_type, group in by_exit_type:
        print(f"\n{exit_type} exits ({len(group)} trades):")
        
        # Count by return level
        at_neg_100 = len(group[abs(group['return_pct'] + 0.001) < 0.0001])
        at_neg_75 = len(group[abs(group['return_pct'] + 0.00075) < 0.0001])
        at_pos_75 = len(group[abs(group['return_pct'] - 0.00075) < 0.0001])
        at_pos_100 = len(group[abs(group['return_pct'] - 0.001) < 0.0001])
        
        if at_neg_100 > 0:
            print(f"  At -0.100%: {at_neg_100} trades")
        if at_neg_75 > 0:
            print(f"  At -0.075%: {at_neg_75} trades")
        if at_pos_75 > 0:
            print(f"  At +0.075%: {at_pos_75} trades")
        if at_pos_100 > 0:
            print(f"  At +0.100%: {at_pos_100} trades")
        
        # Show mean return
        print(f"  Mean return: {group['return_pct'].mean():.4%}")
    
    print("\n" + "="*60 + "\n")
    
    # The smoking gun analysis
    print("=== CRITICAL FINDINGS ===\n")
    
    # Find take_profit exits at -0.1%
    tp_at_loss = trades_df[
        (abs(trades_df['return_pct'] + 0.001) < 0.0001) & 
        (trades_df['exit_type'] == 'take_profit')
    ]
    
    # Find stop_loss exits at +0.075%
    sl_at_gain = trades_df[
        (abs(trades_df['return_pct'] - 0.00075) < 0.0001) & 
        (trades_df['exit_type'] == 'stop_loss')
    ]
    
    if len(tp_at_loss) > 0:
        print(f"❌ Found {len(tp_at_loss)} trades hitting TAKE PROFIT at -0.1% LOSS")
        print("   This indicates inverted stop/target logic")
        
    if len(sl_at_gain) > 0:
        print(f"❌ Found {len(sl_at_gain)} trades hitting STOP LOSS at +0.075% GAIN")
        print("   This indicates inverted stop/target logic")
    
    # Try to determine which direction is affected
    print("\n=== Attempting to identify affected direction ===")
    
    # Look at the actual price movements
    if len(tp_at_loss) > 0:
        print("\nExamining 'take_profit' trades that lost money:")
        for idx in tp_at_loss.index[:3]:  # First 3 examples
            trade = tp_at_loss.loc[idx]
            price_change = trade['exit_price'] - trade['entry_price']
            print(f"  Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f}")
            print(f"  Price moved: ${price_change:.2f} ({price_change/trade['entry_price']*100:.3f}%)")
            if price_change < 0:
                print(f"  → Price went DOWN, so this was likely a LONG position")
            else:
                print(f"  → Price went UP, so this was likely a SHORT position")
    
    return trades_df

# Also check quantity setting
def check_position_sizing():
    """Check where quantity of 100 might be coming from"""
    print("\n=== Position Sizing Check ===\n")
    print("The code shows position_size should be 1, but trades show quantity=100")
    print("\nPossible sources:")
    print("1. Configuration override in YAML file")
    print("2. Strategy-specific position sizing")
    print("3. Risk manager override")
    print("4. Multiplier applied somewhere in the pipeline")
    print("\nTo check config: look for 'position_size' or 'quantity' in your YAML")

# Run both analyses
print("Running analysis...\n")
result = analyze_metadata_deep(df, trades_df)
check_position_sizing()