#!/usr/bin/env python3
"""Check the exact exit prices being used."""

import pandas as pd
from pathlib import Path

print("=== Checking Exact Exit Prices ===")

results_dir = Path("config/bollinger/results/latest")

# Load position closes  
pos_close_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"
if pos_close_file.exists():
    closes = pd.read_parquet(pos_close_file)
    
    # Calculate what the exit prices SHOULD be
    closes['expected_sl_price'] = closes['entry_price'] * 0.99925  # -0.075%
    closes['expected_tp_price'] = closes['entry_price'] * 1.001    # +0.1%
    
    # Check stop losses
    sl_trades = closes[closes['exit_type'] == 'stop_loss'].head(10)
    if len(sl_trades) > 0:
        print("Stop Loss Trades:")
        print("-" * 80)
        for _, trade in sl_trades.iterrows():
            diff = trade['exit_price'] - trade['expected_sl_price']
            print(f"Entry: ${trade['entry_price']:.4f}")
            print(f"  Expected SL: ${trade['expected_sl_price']:.4f}")
            print(f"  Actual Exit: ${trade['exit_price']:.4f}")
            print(f"  Difference: ${diff:.4f}")
            print(f"  Return: {((trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100):.4f}%")
            print()
    
    # Check take profits
    tp_trades = closes[closes['exit_type'] == 'take_profit'].head(10)
    if len(tp_trades) > 0:
        print("\nTake Profit Trades:")
        print("-" * 80)
        for _, trade in tp_trades.iterrows():
            diff = trade['exit_price'] - trade['expected_tp_price']
            print(f"Entry: ${trade['entry_price']:.4f}")
            print(f"  Expected TP: ${trade['expected_tp_price']:.4f}")
            print(f"  Actual Exit: ${trade['exit_price']:.4f}")
            print(f"  Difference: ${diff:.4f}")
            print(f"  Return: {((trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100):.4f}%")
            print()

print("\n=== Key Question ===")
print("Are the actual exit prices close to the expected SL/TP prices?")
print("If not, the execution engine might be using the wrong price.")