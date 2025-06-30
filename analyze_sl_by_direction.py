#!/usr/bin/env python3
"""Analyze stop losses by position direction (long/short)."""

import pandas as pd
from pathlib import Path

print("=== Analyzing Stop Losses by Direction ===")

results_dir = Path("config/bollinger/results/latest")

# Load position opens and closes
pos_open_file = results_dir / "traces/portfolio/positions_open/positions_open.parquet"
pos_close_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"

if pos_open_file.exists() and pos_close_file.exists():
    opens = pd.read_parquet(pos_open_file)
    closes = pd.read_parquet(pos_close_file)
    
    # Merge to get quantity info (positive = long, negative = short)
    # Assuming they're in the same order
    if len(opens) == len(closes):
        closes['quantity'] = opens['metadata'].apply(lambda x: x.get('quantity', 0) if isinstance(x, dict) else 0)
        
        # Parse metadata if it's JSON string
        import json
        for i in range(len(closes)):
            if isinstance(opens.iloc[i]['metadata'], str):
                try:
                    meta = json.loads(opens.iloc[i]['metadata'])
                    closes.loc[closes.index[i], 'quantity'] = meta.get('quantity', 0)
                except:
                    pass
    
    # Calculate returns
    closes['return_pct'] = ((closes['exit_price'] - closes['entry_price']) / closes['entry_price']) * 100
    
    # Identify long vs short positions
    closes['position_type'] = closes['quantity'].apply(lambda x: 'LONG' if x > 0 else 'SHORT' if x < 0 else 'UNKNOWN')
    
    # Analyze stop losses by direction
    sl_trades = closes[closes['exit_type'] == 'stop_loss']
    
    print(f"\nTotal stop loss trades: {len(sl_trades)}")
    
    # Long positions
    long_sl = sl_trades[sl_trades['position_type'] == 'LONG']
    if len(long_sl) > 0:
        print(f"\nLONG positions with stop loss: {len(long_sl)}")
        print(f"  Average return: {long_sl['return_pct'].mean():.4f}%")
        print(f"  Expected return: -0.075%")
        
        # Count gains vs losses
        long_gains = long_sl[long_sl['return_pct'] > 0]
        long_losses = long_sl[long_sl['return_pct'] < 0]
        print(f"  With gains: {len(long_gains)} ({len(long_gains)/len(long_sl)*100:.1f}%)")
        print(f"  With losses: {len(long_losses)} ({len(long_losses)/len(long_sl)*100:.1f}%)")
        
        # Show examples
        print("\n  Examples of LONG stop losses:")
        for i, (_, trade) in enumerate(long_sl.head(5).iterrows()):
            print(f"    Entry: ${trade['entry_price']:.4f}, Exit: ${trade['exit_price']:.4f}, Return: {trade['return_pct']:.4f}%")
    
    # Short positions
    short_sl = sl_trades[sl_trades['position_type'] == 'SHORT']
    if len(short_sl) > 0:
        print(f"\nSHORT positions with stop loss: {len(short_sl)}")
        print(f"  Average return: {short_sl['return_pct'].mean():.4f}%")
        print(f"  Expected return: -0.075%")
        
        # Count gains vs losses
        short_gains = short_sl[short_sl['return_pct'] > 0]
        short_losses = short_sl[short_sl['return_pct'] < 0]
        print(f"  With gains: {len(short_gains)} ({len(short_gains)/len(short_sl)*100:.1f}%)")
        print(f"  With losses: {len(short_losses)} ({len(short_losses)/len(short_sl)*100:.1f}%)")
        
        # Show examples
        print("\n  Examples of SHORT stop losses:")
        for i, (_, trade) in enumerate(short_sl.head(5).iterrows()):
            print(f"    Entry: ${trade['entry_price']:.4f}, Exit: ${trade['exit_price']:.4f}, Return: {trade['return_pct']:.4f}%")
    
    # Check if pattern matches
    print("\n\n=== Analysis ===")
    print("For LONG positions:")
    print("  - Stop loss should trigger when price drops 0.075%")
    print("  - Exit price should be LOWER than entry price")
    print("  - Return should be -0.075%")
    
    print("\nFor SHORT positions:")
    print("  - Stop loss should trigger when price rises 0.075%")
    print("  - Exit price should be HIGHER than entry price")
    print("  - Return should be -0.075%")
    
    print("\n⚠️ If SHORT positions show +0.075% returns, the calculation is inverted!")

# Also check signals to understand position distribution
signals_file = results_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"
if signals_file.exists():
    signals = pd.read_parquet(signals_file)
    
    # Count signal directions
    signal_counts = signals['val'].value_counts()
    print(f"\n\nSignal distribution:")
    print(f"  Long signals (1): {signal_counts.get(1, 0)}")
    print(f"  Short signals (-1): {signal_counts.get(-1, 0)}")
    print(f"  Flat signals (0): {signal_counts.get(0, 0)}")