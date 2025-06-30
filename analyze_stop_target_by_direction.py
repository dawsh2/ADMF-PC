#!/usr/bin/env python3
"""
Analyze stop/target exits by direction to find the inverted logic
"""
import pandas as pd
import json

def analyze_stop_target_by_direction(df, trades_df):
    """Find which direction has inverted stop/target logic"""
    
    print("=== Analyzing Stop/Target Exits by Direction ===\n")
    
    # First, get all stop_loss and take_profit exits
    stop_loss_trades = trades_df[trades_df['exit_type'] == 'stop_loss']
    take_profit_trades = trades_df[trades_df['exit_type'] == 'take_profit']
    
    print(f"Total stop_loss exits: {len(stop_loss_trades)}")
    print(f"Total take_profit exits: {len(take_profit_trades)}")
    print()
    
    # Extract direction for each trade from metadata
    def get_direction(trade):
        # Find matching metadata entry
        matches = df[
            (abs(df['entry_price'] - trade['entry_price']) < 0.01) &
            (abs(df['exit_price'] - trade['exit_price']) < 0.01)
        ]
        
        if len(matches) > 0:
            metadata = matches.iloc[0]['metadata']
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            quantity = metadata.get('quantity', 0)
            return 'LONG' if quantity > 0 else 'SHORT' if quantity < 0 else 'UNKNOWN'
        return 'UNKNOWN'
    
    # Analyze stop_loss exits
    print("=== Stop Loss Exits ===")
    sl_by_direction = {'LONG': [], 'SHORT': []}
    
    for idx, trade in stop_loss_trades.iterrows():
        direction = get_direction(trade)
        if direction in sl_by_direction:
            sl_by_direction[direction].append(trade['return_pct'])
    
    for direction, returns in sl_by_direction.items():
        if returns:
            print(f"\n{direction} stop losses ({len(returns)} trades):")
            print(f"  Mean return: {pd.Series(returns).mean():.4f}%")
            print(f"  Min return: {pd.Series(returns).min():.4f}%")
            print(f"  Max return: {pd.Series(returns).max():.4f}%")
            
            # Check for positive stop losses (wrong!)
            positive_stops = [r for r in returns if r > 0]
            if positive_stops:
                print(f"  ❌ {len(positive_stops)} stop losses with POSITIVE returns!")
                print(f"     These should be negative (losses)")
    
    # Analyze take_profit exits
    print("\n=== Take Profit Exits ===")
    tp_by_direction = {'LONG': [], 'SHORT': []}
    
    for idx, trade in take_profit_trades.iterrows():
        direction = get_direction(trade)
        if direction in tp_by_direction:
            tp_by_direction[direction].append(trade['return_pct'])
    
    for direction, returns in tp_by_direction.items():
        if returns:
            print(f"\n{direction} take profits ({len(returns)} trades):")
            print(f"  Mean return: {pd.Series(returns).mean():.4f}%")
            print(f"  Min return: {pd.Series(returns).min():.4f}%")
            print(f"  Max return: {pd.Series(returns).max():.4f}%")
            
            # Check for negative take profits (wrong!)
            negative_tps = [r for r in returns if r < 0]
            if negative_tps:
                print(f"  ❌ {len(negative_tps)} take profits with NEGATIVE returns!")
                print(f"     These should be positive (profits)")
                
                # Show examples
                print(f"\n  Examples of {direction} take profits at losses:")
                examples = take_profit_trades[take_profit_trades['return_pct'] < 0].head(3)
                for idx, ex in examples.iterrows():
                    if get_direction(ex) == direction:
                        print(f"    Entry: ${ex['entry_price']:.2f} → Exit: ${ex['exit_price']:.2f}")
                        print(f"    Return: {ex['return_pct']:.4f}%")
    
    # Final verdict
    print("\n" + "="*60)
    print("\n=== FINAL VERDICT ===\n")
    
    # Check which direction has inverted logic
    long_tp_negative = len([r for r in tp_by_direction.get('LONG', []) if r < 0])
    short_tp_negative = len([r for r in tp_by_direction.get('SHORT', []) if r < 0])
    
    long_sl_positive = len([r for r in sl_by_direction.get('LONG', []) if r > 0])
    short_sl_positive = len([r for r in sl_by_direction.get('SHORT', []) if r > 0])
    
    if long_tp_negative > 0:
        print(f"❌ LONG positions have INVERTED take profit logic")
        print(f"   {long_tp_negative} LONG trades hit take_profit with LOSSES")
    
    if short_tp_negative > 0:
        print(f"❌ SHORT positions have INVERTED take profit logic")
        print(f"   {short_tp_negative} SHORT trades hit take_profit with LOSSES")
        
    if long_sl_positive > 0:
        print(f"❌ LONG positions have INVERTED stop loss logic")
        print(f"   {long_sl_positive} LONG trades hit stop_loss with GAINS")
        
    if short_sl_positive > 0:
        print(f"❌ SHORT positions have INVERTED stop loss logic")
        print(f"   {short_sl_positive} SHORT trades hit stop_loss with GAINS")

# Run the analysis
analyze_stop_target_by_direction(df, trades_df)