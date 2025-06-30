#!/usr/bin/env python3
"""Debug why we have 463 trades instead of 416."""

import pandas as pd
from pathlib import Path

print("=== Debugging 463 Trades Issue ===")

results_dir = Path("config/bollinger/results/latest")

# Load trades
trades_file = results_dir / "traces/events/portfolio/trades.parquet"
if trades_file.exists():
    trades = pd.read_parquet(trades_file)
    print(f"\nTotal trades: {len(trades)}")
    
    # Check exit types
    print("\nExit type breakdown:")
    print(trades['exit_type'].value_counts())
    
    # Check if we have strategy_id
    if 'strategy_id' in trades.columns:
        missing_strategy = trades['strategy_id'].isna().sum()
        print(f"\nTrades missing strategy_id: {missing_strategy}")
    else:
        print("\n❌ No strategy_id column in trades!")
    
    # Calculate returns
    trades['return_pct'] = ((trades['exit_price'] - trades['entry_price']) / trades['entry_price']) * 100
    
    # Look for stop losses that are gains
    stop_loss_trades = trades[trades['exit_type'] == 'stop_loss']
    if len(stop_loss_trades) > 0:
        sl_gains = stop_loss_trades[stop_loss_trades['return_pct'] > 0]
        print(f"\n⚠️ Stop loss trades with GAINS: {len(sl_gains)} out of {len(stop_loss_trades)}")
        
        if len(sl_gains) > 0:
            print("\nFirst 5 stop losses with gains:")
            for i, (_, trade) in enumerate(sl_gains.head().iterrows()):
                print(f"{i+1}. {trade['timestamp']}: Return={trade['return_pct']:.4f}%")
    
    # Check for immediate re-entries
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    trades = trades.sort_values(['symbol', 'timestamp'])
    
    immediate_reentries = 0
    risk_exit_types = ['stop_loss', 'take_profit', 'trailing_stop']
    
    for symbol in trades['symbol'].unique():
        symbol_trades = trades[trades['symbol'] == symbol].reset_index(drop=True)
        
        for i in range(len(symbol_trades) - 1):
            curr = symbol_trades.iloc[i]
            next = symbol_trades.iloc[i + 1]
            
            if curr.get('exit_type') in risk_exit_types:
                time_diff = (next['timestamp'] - curr['timestamp']).total_seconds() / 60
                if time_diff == 0:
                    immediate_reentries += 1
    
    print(f"\nImmediate re-entries after risk exits: {immediate_reentries}")
    
    # Check signal events to see if OHLC is there
    signal_events_file = results_dir / "traces/events/strategy/signal_events.parquet"
    if signal_events_file.exists():
        signals = pd.read_parquet(signal_events_file)
        print(f"\n\nChecking signal structure...")
        
        # Check first signal's metadata
        if not signals.empty and 'metadata' in signals.columns:
            first_metadata = signals.iloc[0]['metadata']
            if isinstance(first_metadata, dict):
                print("Signal metadata keys:", list(first_metadata.keys()))
                has_ohlc = all(k in first_metadata for k in ['open', 'high', 'low', 'close'])
                print(f"Has OHLC data: {has_ohlc}")
                
                if 'high' in first_metadata and 'low' in first_metadata:
                    print(f"Sample: high={first_metadata['high']}, low={first_metadata['low']}")
else:
    print("Trades file not found!")

print("\n\n=== Analysis ===")
print("If you're seeing 463 trades (up from 453), possible causes:")
print("1. OHLC data is now causing MORE exits (both SL and TP hitting in same bar)")
print("2. Exit memory still not working (check strategy_id)")
print("3. Stop losses still exiting at wrong prices")