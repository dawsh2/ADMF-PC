#!/usr/bin/env python3
"""
Extract signal information from metadata
"""
import pandas as pd
import json

def analyze_signals_from_metadata(df, trades_df):
    """Extract and analyze signal directions from metadata"""
    
    print("=== Extracting Signal Information from Metadata ===\n")
    
    # First, let's examine the metadata structure
    print("Sample metadata:")
    for i in range(min(3, len(df))):
        metadata = df.iloc[i]['metadata']
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        print(f"\nRow {i}:")
        print(f"  Exit type: {df.iloc[i]['exit_type']}")
        print(f"  Metadata keys: {list(metadata.keys())}")
        if 'signal' in metadata:
            print(f"  Signal: {metadata['signal']}")
        if 'direction' in metadata:
            print(f"  Direction: {metadata['direction']}")
        if 'side' in metadata:
            print(f"  Side: {metadata['side']}")
    
    # Extract signal information for all trades
    trades_with_signals = []
    
    for idx, trade in trades_df.iterrows():
        # Find the corresponding exit event
        exit_events = df[
            (df['entry_price'] == trade['entry_price']) &
            (df['exit_price'] == trade['exit_price'])
        ]
        
        if len(exit_events) > 0:
            metadata = exit_events.iloc[0]['metadata']
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            trade_dict = trade.to_dict()
            
            # Extract signal/direction info
            if 'signal' in metadata:
                signal_val = metadata['signal']
                trade_dict['signal_value'] = signal_val
                trade_dict['signal_direction'] = 'LONG' if signal_val > 0 else 'SHORT' if signal_val < 0 else 'FLAT'
            elif 'direction' in metadata:
                trade_dict['signal_direction'] = metadata['direction']
            elif 'side' in metadata:
                # BUY = LONG, SELL = SHORT
                trade_dict['signal_direction'] = 'LONG' if metadata['side'] == 'BUY' else 'SHORT'
            
            trades_with_signals.append(trade_dict)
    
    trades_analysis = pd.DataFrame(trades_with_signals)
    
    # Analyze by direction
    print("\n\n=== Analysis by Signal Direction ===")
    
    for direction in ['LONG', 'SHORT']:
        dir_trades = trades_analysis[trades_analysis['signal_direction'] == direction]
        
        if len(dir_trades) > 0:
            print(f"\n{direction} positions ({len(dir_trades)} trades):")
            
            # Count exits at each level
            neg_100 = dir_trades[abs(dir_trades['return_pct'] + 0.001) < 0.0001]
            neg_75 = dir_trades[abs(dir_trades['return_pct'] + 0.00075) < 0.0001]
            pos_75 = dir_trades[abs(dir_trades['return_pct'] - 0.00075) < 0.0001]
            pos_100 = dir_trades[abs(dir_trades['return_pct'] - 0.001) < 0.0001]
            
            print(f"  Exits at -0.100%: {len(neg_100)} trades")
            print(f"  Exits at -0.075%: {len(neg_75)} trades")
            print(f"  Exits at +0.075%: {len(pos_75)} trades")
            print(f"  Exits at +0.100%: {len(pos_100)} trades")
            
            # Check exit types at each level
            if len(neg_100) > 0:
                exit_types = neg_100['exit_type'].value_counts()
                print(f"\n  At -0.100% loss:")
                for exit_type, count in exit_types.items():
                    print(f"    {exit_type}: {count}")
                    if exit_type == 'take_profit':
                        print(f"      ⚠️ {direction} hitting TAKE PROFIT at a LOSS!")
            
            if len(pos_75) > 0:
                exit_types = pos_75['exit_type'].value_counts()
                print(f"\n  At +0.075% gain:")
                for exit_type, count in exit_types.items():
                    print(f"    {exit_type}: {count}")
                    if exit_type == 'stop_loss':
                        print(f"      ⚠️ {direction} hitting STOP LOSS at a GAIN!")
    
    # Summary
    print("\n\n=== DIAGNOSIS ===")
    
    # Count problematic exits
    long_trades = trades_analysis[trades_analysis['signal_direction'] == 'LONG']
    short_trades = trades_analysis[trades_analysis['signal_direction'] == 'SHORT']
    
    long_inverted = len(long_trades[
        (abs(long_trades['return_pct'] + 0.001) < 0.0001) & 
        (long_trades['exit_type'] == 'take_profit')
    ])
    
    short_inverted = len(short_trades[
        (abs(short_trades['return_pct'] + 0.001) < 0.0001) & 
        (short_trades['exit_type'] == 'take_profit')
    ])
    
    if long_inverted > 0 and short_inverted == 0:
        print("❌ LONG positions have INVERTED stop/target logic!")
        print(f"   {long_inverted} long trades hit 'take_profit' at -0.1% LOSS")
        print("   SHORT positions appear to be working correctly")
    elif short_inverted > 0 and long_inverted == 0:
        print("❌ SHORT positions have INVERTED stop/target logic!")
        print(f"   {short_inverted} short trades hit 'take_profit' at -0.1% LOSS")
        print("   LONG positions appear to be working correctly")
    elif long_inverted > 0 and short_inverted > 0:
        print("❌ BOTH directions have issues!")
        print(f"   {long_inverted} LONG trades hit 'take_profit' at -0.1% LOSS")
        print(f"   {short_inverted} SHORT trades hit 'take_profit' at -0.1% LOSS")
    
    return trades_analysis

# Run the analysis
result = analyze_signals_from_metadata(df, trades_df)