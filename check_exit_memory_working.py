#!/usr/bin/env python3
"""Check if exit memory is working in the latest results."""

import pandas as pd
import json
from pathlib import Path

print("=== Checking Exit Memory in Latest Results ===")

results_dir = Path("/Users/daws/ADMF-PC/config/bollinger/results/20250627_145037")

# Load position events
pos_open_file = results_dir / "traces/portfolio/positions_open/positions_open.parquet"
pos_close_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"

if pos_open_file.exists() and pos_close_file.exists():
    opens = pd.read_parquet(pos_open_file)
    closes = pd.read_parquet(pos_close_file)
    
    print(f"Found {len(opens)} position opens and {len(closes)} position closes")
    
    # Parse metadata
    for df in [opens, closes]:
        if 'metadata' in df.columns:
            for i in range(len(df)):
                if isinstance(df.iloc[i]['metadata'], str):
                    try:
                        meta = json.loads(df.iloc[i]['metadata'])
                        for key, value in meta.items():
                            if key not in df.columns:
                                df.loc[df.index[i], key] = value
                    except:
                        pass
    
    # Check for immediate re-entries after risk exits
    immediate_reentries = []
    
    for i in range(min(len(opens), len(closes)) - 1):
        close_event = closes.iloc[i]
        exit_type = close_event.get('exit_type', 'unknown')
        
        # Only check risk exits
        if exit_type in ['stop_loss', 'take_profit', 'trailing_stop']:
            next_open = opens.iloc[i + 1]
            bars_between = next_open['idx'] - close_event['idx']
            
            if bars_between <= 1:  # Immediate re-entry
                immediate_reentries.append({
                    'close_idx': i,
                    'close_bar': close_event['idx'],
                    'exit_type': exit_type,
                    'next_open_bar': next_open['idx'],
                    'bars_between': bars_between
                })
    
    print(f"\nImmediate re-entries after risk exits: {len(immediate_reentries)}")
    
    if immediate_reentries:
        print("\nFirst 5 examples:")
        for i, reentry in enumerate(immediate_reentries[:5]):
            print(f"{i+1}. {reentry['exit_type']} at bar {reentry['close_bar']}, re-entry at bar {reentry['next_open_bar']} (gap: {reentry['bars_between']})")
    
    # Check exit type distribution
    if 'exit_type' in closes.columns:
        print("\nExit type distribution:")
        print(closes['exit_type'].value_counts())
        
        # Count risk exits
        risk_exits = closes[closes['exit_type'].isin(['stop_loss', 'take_profit', 'trailing_stop'])]
        print(f"\nTotal risk exits: {len(risk_exits)}")
        
    # Check if we're still getting 463 trades
    print(f"\n=== Trade Count ===")
    print(f"Expected (from notebook): 463")
    print(f"Actual: {len(opens)}")
    
    if len(opens) == 463:
        print("\n⚠️ Still seeing 463 trades - exit memory fix may not be applied!")
        print("Make sure to:")
        print("1. Clear Python cache: rm -rf __pycache__ src/**/__pycache__")
        print("2. Restart Python/Jupyter kernel")
        print("3. Re-run the backtest")
        
else:
    print("Could not find position event files")