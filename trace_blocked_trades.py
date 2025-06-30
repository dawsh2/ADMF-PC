#!/usr/bin/env python3
"""Trace why trades are being blocked."""

import pandas as pd
import json
from pathlib import Path

print("=== Tracing Blocked Trades ===")

results_dir = Path("config/bollinger/results/latest")

# Load all data
opens_file = results_dir / "traces/portfolio/positions_open/positions_open.parquet"
closes_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"
signals_file = results_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"

if all(f.exists() for f in [opens_file, closes_file, signals_file]):
    opens = pd.read_parquet(opens_file)
    closes = pd.read_parquet(closes_file)
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
    
    print(f"Trades executed: {len(opens)}")
    print(f"Entry signals in data: 415")
    
    # Get the bars where trades actually happened
    trade_bars = set(opens['idx'].values)
    
    # Get all entry signal bars
    entry_transitions = []
    for i in range(1, len(signals)):
        if signals.iloc[i-1]['val'] == 0 and signals.iloc[i]['val'] != 0:
            entry_transitions.append(signals.iloc[i]['idx'])
    
    print(f"\nEntry signal bars: {len(entry_transitions)}")
    print(f"Bars with trades: {len(trade_bars)}")
    
    # Find which entry signals didn't result in trades
    blocked_entries = [bar for bar in entry_transitions if bar not in trade_bars]
    print(f"Blocked entry signals: {len(blocked_entries)}")
    
    # Analyze pattern
    print("\n=== Trade Pattern ===")
    print("Executed trades at bars:", sorted(trade_bars))
    
    print("\n=== First Risk Exit Analysis ===")
    if len(closes) > 0:
        first_close = closes.iloc[0]
        print(f"First close: {first_close['exit_type']} at bar {first_close['idx']}")
        
        # Find all entry signals after this exit
        signals_after_exit = [bar for bar in entry_transitions if bar > first_close['idx']]
        print(f"Entry signals after first exit: {len(signals_after_exit)}")
        
        # How many resulted in trades?
        trades_after_exit = [bar for bar in signals_after_exit if bar in trade_bars]
        print(f"Trades after first exit: {len(trades_after_exit)}")
        
        if len(trades_after_exit) == 0:
            print("\n⚠️ NO TRADES after first risk exit!")
            print("Exit memory is likely stuck and never clearing.")
    
    # Check if exit memory is getting stuck
    print("\n=== Checking Exit Memory Behavior ===")
    
    # For each risk exit, check subsequent signals
    risk_exits = closes[closes['exit_type'].isin(['stop_loss', 'take_profit'])]
    
    for i, exit in risk_exits.iterrows():
        exit_bar = exit['idx']
        exit_type = exit['exit_type']
        
        # Get signal at time of exit
        signal_at_exit = signals[signals['idx'] <= exit_bar].iloc[-1]['val'] if len(signals[signals['idx'] <= exit_bar]) > 0 else None
        
        print(f"\n{exit_type} at bar {exit_bar}, signal was: {signal_at_exit}")
        
        # Check next few signals
        next_signals = signals[signals['idx'] > exit_bar].head(10)
        different_signals = next_signals[next_signals['val'] != signal_at_exit]
        
        if len(different_signals) > 0:
            first_different = different_signals.iloc[0]
            print(f"  Signal changes to {first_different['val']} at bar {first_different['idx']} (+{first_different['idx'] - exit_bar} bars)")
            
            # Did we get a trade after signal changed?
            next_trade = opens[opens['idx'] > first_different['idx']].head(1)
            if len(next_trade) > 0:
                print(f"  Next trade at bar {next_trade.iloc[0]['idx']} ✓")
            else:
                print(f"  No trades after signal change ❌")
    
    print("\n=== Hypothesis ===")
    print("Exit memory might be getting 'stuck' if:")
    print("1. The stored signal value doesn't match how we're comparing")
    print("2. The signal tracking in portfolio is incorrect")
    print("3. The sparse signal storage is causing comparison issues")