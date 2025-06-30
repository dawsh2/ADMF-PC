#!/usr/bin/env python3
"""Trace signal flow to understand exit memory failures."""

import pandas as pd
import json
from pathlib import Path

print("=== Tracing Signal Flow ===")

results_dir = Path("config/bollinger/results/latest")

# Load data
pos_open_file = results_dir / "traces/portfolio/positions_open/positions_open.parquet"
pos_close_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"
signals_file = results_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"

if all(f.exists() for f in [pos_open_file, pos_close_file, signals_file]):
    opens = pd.read_parquet(pos_open_file)
    closes = pd.read_parquet(pos_close_file)
    signals = pd.read_parquet(signals_file)
    
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
    
    # Find a specific case to debug
    print("\n=== Finding Risk Exit with Immediate Re-entry ===")
    
    for i in range(min(len(opens), len(closes)) - 1):
        close_event = closes.iloc[i]
        exit_type = close_event.get('exit_type', 'unknown')
        
        if exit_type in ['stop_loss', 'take_profit']:
            next_open = opens.iloc[i + 1]
            bars_between = next_open['idx'] - close_event['idx']
            
            if bars_between <= 1:
                # Found one - trace it
                exit_bar = close_event['idx']
                open_bar = next_open['idx']
                
                print(f"\nCase: {exit_type} at bar {exit_bar}, re-entry at bar {open_bar}")
                
                # Get signals around this event
                start_bar = exit_bar - 5
                end_bar = open_bar + 5
                
                print(f"\nSignals from bar {start_bar} to {end_bar}:")
                
                for bar in range(start_bar, end_bar + 1):
                    sig = signals[signals['idx'] == bar]
                    if len(sig) > 0:
                        print(f"  Bar {bar}: signal={sig.iloc[0]['val']}")
                    else:
                        print(f"  Bar {bar}: NO SIGNAL DATA")
                    
                    if bar == exit_bar:
                        print(f"     ^^^ EXIT HERE ({exit_type})")
                    if bar == open_bar:
                        print(f"     ^^^ RE-ENTRY HERE")
                
                # Find last signal before exit
                signals_before = signals[signals['idx'] <= exit_bar].sort_values('idx')
                if len(signals_before) > 0:
                    last_signal = signals_before.iloc[-1]
                    print(f"\nLast signal before exit: {last_signal['val']} at bar {last_signal['idx']}")
                
                break
    
    # Check signal values
    print("\n=== Signal Values ===")
    print(f"Unique signals: {sorted(signals['val'].unique())}")

else:
    print("Could not find required files")