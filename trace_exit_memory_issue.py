#!/usr/bin/env python3
"""Trace exit memory issue - why are immediate re-entries happening?"""

import pandas as pd
import json
from pathlib import Path

print("=== Tracing Exit Memory Issue ===")

results_dir = Path("config/bollinger/results/latest")

# Load all required data
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
    
    # Find immediate re-entries after risk exits
    print("\n=== Analyzing Immediate Re-entries ===")
    
    immediate_problems = []
    
    for i in range(min(len(opens), len(closes)) - 1):
        close_event = closes.iloc[i]
        exit_type = close_event.get('exit_type', 'unknown')
        
        # Only check risk exits
        if exit_type in ['stop_loss', 'take_profit', 'trailing_stop']:
            next_open = opens.iloc[i + 1]
            bars_between = next_open['idx'] - close_event['idx']
            
            if bars_between <= 1:  # Immediate re-entry
                # Get signal values around the exit
                exit_bar = close_event['idx']
                signal_at_exit = signals[signals['idx'] == exit_bar]
                signal_at_open = signals[signals['idx'] == next_open['idx']]
                
                problem = {
                    'close_bar': exit_bar,
                    'exit_type': exit_type,
                    'open_bar': next_open['idx'],
                    'bars_between': bars_between,
                    'close_strategy_id': close_event.get('strategy_id', 'unknown'),
                    'open_strategy_id': next_open.get('strategy_id', 'unknown'),
                    'signal_at_exit': signal_at_exit.iloc[0]['val'] if len(signal_at_exit) > 0 else None,
                    'signal_at_open': signal_at_open.iloc[0]['val'] if len(signal_at_open) > 0 else None
                }
                immediate_problems.append(problem)
    
    print(f"Found {len(immediate_problems)} immediate re-entries after risk exits")
    
    # Analyze the problems
    if immediate_problems:
        # Check patterns
        print("\n=== Pattern Analysis ===")
        
        # 1. Are strategy_ids being tracked?
        has_close_id = sum(1 for p in immediate_problems if p['close_strategy_id'] != 'unknown')
        has_open_id = sum(1 for p in immediate_problems if p['open_strategy_id'] != 'unknown') 
        print(f"Closes with strategy_id: {has_close_id}/{len(immediate_problems)}")
        print(f"Opens with strategy_id: {has_open_id}/{len(immediate_problems)}")
        
        # 2. Are signals persisting?
        signal_persists = sum(1 for p in immediate_problems 
                             if p['signal_at_exit'] == p['signal_at_open'] 
                             and p['signal_at_exit'] is not None 
                             and p['signal_at_exit'] != 0)
        print(f"Signal persists after exit: {signal_persists}/{len(immediate_problems)}")
        
        # 3. Show detailed examples
        print("\n=== Detailed Examples (first 5) ===")
        for i, problem in enumerate(immediate_problems[:5]):
            print(f"\n{i+1}. {problem['exit_type']} at bar {problem['close_bar']}")
            print(f"   Close strategy_id: {problem['close_strategy_id']}")
            print(f"   Signal at exit: {problem['signal_at_exit']}")
            print(f"   Re-open at bar {problem['open_bar']} (gap: {problem['bars_between']})")
            print(f"   Open strategy_id: {problem['open_strategy_id']}")
            print(f"   Signal at open: {problem['signal_at_open']}")
            
            # Check if this should have been blocked
            if problem['signal_at_exit'] == problem['signal_at_open'] and problem['signal_at_exit'] != 0:
                print(f"   ⚠️ SHOULD HAVE BEEN BLOCKED BY EXIT MEMORY!")
    
    # Check sparse signal storage
    print("\n=== Signal Storage Analysis ===")
    print(f"Total signals stored: {len(signals)}")
    print(f"Signal range: bars {signals['idx'].min()} to {signals['idx'].max()}")
    
    # Check for signal gaps
    signal_bars = set(signals['idx'].values)
    all_bars = set(range(signals['idx'].min(), signals['idx'].max() + 1))
    missing_bars = all_bars - signal_bars
    print(f"Missing signal bars: {len(missing_bars)}")
    
    if missing_bars and immediate_problems:
        # Check if any immediate re-entries happened in missing bars
        missing_reentries = 0
        for problem in immediate_problems:
            if problem['open_bar'] in missing_bars:
                missing_reentries += 1
        print(f"Re-entries at missing signal bars: {missing_reentries}")
        
        if missing_reentries > 0:
            print("\n⚠️ ISSUE: Re-entries happening at bars with no signal data!")
            print("This could mean exit memory can't check the signal value.")
    
else:
    print("Could not find required files")