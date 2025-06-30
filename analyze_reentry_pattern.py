#!/usr/bin/env python3
"""Analyze the re-entry pattern in detail."""

import pandas as pd
import json
from pathlib import Path

print("=== Analyzing Re-entry Pattern ===")

results_dir = Path("config/bollinger/results/latest")

# Load all data
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
    
    print(f"Total trades: {len(opens)}")
    
    # Find immediate re-entries after risk exits
    immediate_reentries = []
    
    for i in range(min(len(opens), len(closes)) - 1):
        close_event = closes.iloc[i]
        exit_type = close_event.get('exit_type', 'unknown')
        
        if exit_type in ['stop_loss', 'take_profit', 'trailing_stop']:
            next_open = opens.iloc[i + 1]
            bars_between = next_open['idx'] - close_event['idx']
            
            if bars_between <= 1:
                immediate_reentries.append({
                    'trade_num': i,
                    'close_bar': close_event['idx'],
                    'exit_type': exit_type,
                    'open_bar': next_open['idx'],
                    'bars_between': bars_between
                })
    
    print(f"Immediate re-entries after risk exits: {len(immediate_reentries)}")
    
    # Analyze first few in detail
    print("\n=== Detailed Analysis of First 3 Re-entries ===")
    
    for idx, reentry in enumerate(immediate_reentries[:3]):
        trade_num = reentry['trade_num']
        close_bar = reentry['close_bar']
        open_bar = reentry['open_bar']
        
        print(f"\n{idx+1}. Trade #{trade_num}: {reentry['exit_type']} at bar {close_bar}")
        
        # Get the close details
        close_event = closes.iloc[trade_num]
        print(f"   Position closed: qty={close_event.get('quantity', 'unknown')}")
        print(f"   Entry price: ${close_event.get('entry_price', 0):.4f}")
        print(f"   Exit price: ${close_event.get('exit_price', 0):.4f}")
        
        # Check signals around the exit
        signal_window = signals[(signals['idx'] >= close_bar - 2) & (signals['idx'] <= open_bar + 2)]
        
        print(f"\n   Signals around exit:")
        for _, sig in signal_window.iterrows():
            marker = ""
            if sig['idx'] == close_bar:
                marker = " <- EXIT HERE"
            elif sig['idx'] == open_bar:
                marker = " <- RE-ENTRY HERE"
            print(f"     Bar {sig['idx']}: signal={sig['val']}{marker}")
        
        # Check if signal pattern matches expected re-entry
        signals_before_exit = signals[signals['idx'] < close_bar].tail(1)
        signal_at_reentry = signals[signals['idx'] == open_bar]
        
        if len(signals_before_exit) > 0:
            last_signal_before = signals_before_exit.iloc[0]['val']
            print(f"\n   Last signal before exit: {last_signal_before}")
            
            if len(signal_at_reentry) > 0:
                reentry_signal = signal_at_reentry.iloc[0]['val']
                print(f"   Signal at re-entry: {reentry_signal}")
                
                if last_signal_before == reentry_signal and last_signal_before != 0:
                    print(f"   ⚠️ SAME SIGNAL! Exit memory should have blocked this!")
                elif reentry_signal == 0:
                    print(f"   Signal is FLAT at re-entry")
                else:
                    print(f"   Signal changed from {last_signal_before} to {reentry_signal}")
    
    # Check overall pattern
    print("\n=== Pattern Summary ===")
    
    # Group by exit type
    by_exit_type = {}
    for reentry in immediate_reentries:
        exit_type = reentry['exit_type']
        if exit_type not in by_exit_type:
            by_exit_type[exit_type] = []
        by_exit_type[exit_type].append(reentry)
    
    for exit_type, reentries in by_exit_type.items():
        print(f"\n{exit_type}: {len(reentries)} immediate re-entries")
        
        # Check if they're all at the same bar gap
        gaps = [r['bars_between'] for r in reentries]
        print(f"  Bar gaps: {set(gaps)}")

else:
    print("Could not find required files")