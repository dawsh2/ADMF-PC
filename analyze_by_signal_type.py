#!/usr/bin/env python3
"""
Analyze trades by extracting signal direction from metadata
"""
import pandas as pd
import json

def analyze_by_signal_type(df, trades_df):
    """Match trades with their signal directions from metadata"""
    
    print("=== Extracting Signal Directions ===\n")
    
    # Extract signal info from metadata
    signal_info = []
    
    for idx, row in df.iterrows():
        metadata = row['metadata']
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        # Get the quantity which indicates direction
        quantity = metadata.get('quantity', 0)
        
        # Determine direction from quantity
        direction = 'LONG' if quantity > 0 else 'SHORT' if quantity < 0 else 'FLAT'
        
        signal_info.append({
            'entry_price': row['entry_price'],
            'exit_price': row['exit_price'],
            'exit_type': row['exit_type'],
            'direction': direction,
            'signed_quantity': quantity,
            'return_pct': (row['exit_price'] / row['entry_price'] - 1)
        })
    
    signals_df = pd.DataFrame(signal_info)
    
    # Now match with trades
    trades_with_direction = []
    
    for idx, trade in trades_df.iterrows():
        # Find matching signal info
        matches = signals_df[
            (abs(signals_df['entry_price'] - trade['entry_price']) < 0.01) &
            (abs(signals_df['exit_price'] - trade['exit_price']) < 0.01)
        ]
        
        if len(matches) > 0:
            trade_dict = trade.to_dict()
            trade_dict['direction'] = matches.iloc[0]['direction']
            trade_dict['signed_quantity'] = matches.iloc[0]['signed_quantity']
            trades_with_direction.append(trade_dict)
    
    trades_dir = pd.DataFrame(trades_with_direction)
    
    print(f"Matched {len(trades_dir)} trades with directions\n")
    
    # Analyze by direction
    for direction in ['LONG', 'SHORT']:
        dir_trades = trades_dir[trades_dir['direction'] == direction]
        
        if len(dir_trades) == 0:
            continue
            
        print(f"\n=== {direction} Positions ({len(dir_trades)} trades) ===")
        
        # Count by return level and exit type
        for return_level, label in [(-0.001, "-0.1%"), (-0.00075, "-0.075%"), 
                                   (0.00075, "+0.075%"), (0.001, "+0.1%")]:
            
            at_level = dir_trades[abs(dir_trades['return_pct'] - return_level) < 0.0001]
            
            if len(at_level) > 0:
                print(f"\nAt {label} return: {len(at_level)} trades")
                
                # Group by exit type
                by_exit = at_level.groupby('exit_type').size()
                for exit_type, count in by_exit.items():
                    print(f"  {exit_type}: {count}")
                    
                    # Flag problematic combinations
                    if return_level == -0.001 and exit_type == 'take_profit':
                        print(f"    ❌ {direction} hit TAKE PROFIT at -0.1% LOSS!")
                    elif return_level == 0.00075 and exit_type == 'stop_loss':
                        print(f"    ❌ {direction} hit STOP LOSS at +0.075% GAIN!")
    
    # Final diagnosis
    print("\n" + "="*60)
    print("\n=== FINAL DIAGNOSIS ===\n")
    
    # Count problematic trades by direction
    for direction in ['LONG', 'SHORT']:
        dir_trades = trades_dir[trades_dir['direction'] == direction]
        
        tp_losses = len(dir_trades[
            (abs(dir_trades['return_pct'] + 0.001) < 0.0001) & 
            (dir_trades['exit_type'] == 'take_profit')
        ])
        
        sl_gains = len(dir_trades[
            (abs(dir_trades['return_pct'] - 0.00075) < 0.0001) & 
            (dir_trades['exit_type'] == 'stop_loss')
        ])
        
        if tp_losses > 0 or sl_gains > 0:
            print(f"❌ {direction} positions have INVERTED stop/target logic:")
            if tp_losses > 0:
                print(f"   - {tp_losses} trades hit take_profit at -0.1% LOSS")
            if sl_gains > 0:
                print(f"   - {sl_gains} trades hit stop_loss at +0.075% GAIN")
    
    return trades_dir

# Run analysis
result = analyze_by_signal_type(df, trades_df)