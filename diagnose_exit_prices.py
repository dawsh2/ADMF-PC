#!/usr/bin/env python3
"""Diagnose why exit prices are wrong."""

import pandas as pd
import numpy as np
from pathlib import Path

print("=== Diagnosing Exit Price Issues ===")

results_dir = Path("config/bollinger/results/latest")

# Load position closes
pos_close_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"
if pos_close_file.exists():
    closes = pd.read_parquet(pos_close_file)
    print(f"\nTotal closed positions: {len(closes)}")
    
    # Calculate returns
    if 'entry_price' in closes.columns and 'exit_price' in closes.columns:
        closes['return_pct'] = ((closes['exit_price'] - closes['entry_price']) / closes['entry_price']) * 100
        
        # Group by exit type
        if 'exit_type' in closes.columns:
            print("\nExit type analysis:")
            for exit_type in ['stop_loss', 'take_profit', 'signal']:
                exits = closes[closes['exit_type'] == exit_type]
                if len(exits) > 0:
                    print(f"\n{exit_type}: {len(exits)} trades")
                    print(f"  Average return: {exits['return_pct'].mean():.4f}%")
                    print(f"  Min return: {exits['return_pct'].min():.4f}%")
                    print(f"  Max return: {exits['return_pct'].max():.4f}%")
                    print(f"  Std dev: {exits['return_pct'].std():.4f}%")
                    
                    # For stop losses, check distribution around -0.075%
                    if exit_type == 'stop_loss':
                        expected_sl = -0.075
                        close_to_expected = exits[abs(exits['return_pct'] - expected_sl) < 0.01]
                        print(f"  Near {expected_sl}%: {len(close_to_expected)} trades")
                        
                        # Check for stop losses with gains
                        sl_with_gains = exits[exits['return_pct'] > 0]
                        print(f"  ⚠️ Stop losses with GAINS: {len(sl_with_gains)}")
                        
                    # For take profits, check distribution around 0.1%
                    elif exit_type == 'take_profit':
                        expected_tp = 0.1
                        close_to_expected = exits[abs(exits['return_pct'] - expected_tp) < 0.01]
                        print(f"  Near {expected_tp}%: {len(close_to_expected)} trades")
                        
                        # Check for take profits with losses
                        tp_with_losses = exits[exits['return_pct'] < 0]
                        print(f"  ⚠️ Take profits with LOSSES: {len(tp_with_losses)}")
        
        # Show some example trades
        print("\n\nExample trades:")
        
        # Stop loss examples
        sl_trades = closes[closes['exit_type'] == 'stop_loss'].head(5)
        if len(sl_trades) > 0:
            print("\nStop loss examples:")
            for i, trade in sl_trades.iterrows():
                print(f"  Entry: ${trade['entry_price']:.4f}, Exit: ${trade['exit_price']:.4f}, Return: {trade['return_pct']:.4f}%")
        
        # Take profit examples
        tp_trades = closes[closes['exit_type'] == 'take_profit'].head(5)
        if len(tp_trades) > 0:
            print("\nTake profit examples:")
            for i, trade in tp_trades.iterrows():
                print(f"  Entry: ${trade['entry_price']:.4f}, Exit: ${trade['exit_price']:.4f}, Return: {trade['return_pct']:.4f}%")

# Check if signals have OHLC data
signals_file = results_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"
if signals_file.exists():
    signals = pd.read_parquet(signals_file)
    print(f"\n\nSignal analysis:")
    print(f"Total signals: {len(signals)}")
    
    # Check metadata
    if len(signals) > 0:
        first = signals.iloc[0]
        print("\nFirst signal metadata type:", type(first.get('metadata', 'N/A')))
        
        if 'metadata' in first and isinstance(first['metadata'], dict):
            print("Metadata keys:", list(first['metadata'].keys()))
            
            # Check for OHLC
            has_ohlc = all(k in first['metadata'] for k in ['open', 'high', 'low', 'close'])
            print(f"Has OHLC data: {has_ohlc}")
            
            if has_ohlc:
                # Check a few signals to see OHLC ranges
                print("\nSample OHLC ranges:")
                for i in range(min(5, len(signals))):
                    sig = signals.iloc[i]
                    meta = sig['metadata']
                    if isinstance(meta, dict) and 'high' in meta and 'low' in meta:
                        range_pct = ((meta['high'] - meta['low']) / meta['low']) * 100
                        print(f"  Bar {i}: Range = {range_pct:.4f}% (H:{meta['high']:.2f}, L:{meta['low']:.2f})")

print("\n\n=== Diagnosis ===")
print("If stop losses average 0.00% and take profits average -0.01%, then:")
print("1. Exit prices are NOT being calculated correctly")
print("2. Might be using entry price or close price instead of calculated exit price")
print("3. OR the high/low data might be incorrect")