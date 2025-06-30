#!/usr/bin/env python3
"""Check signal persistence and sparse storage issue."""

import pandas as pd
from pathlib import Path

print("=== Checking Signal Persistence ===")

results_dir = Path("config/bollinger/results/latest")
signals_file = results_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"

if signals_file.exists():
    signals = pd.read_parquet(signals_file)
    
    print(f"Total signals: {len(signals)}")
    print(f"Signal values: {signals['val'].value_counts()}")
    
    # Check how sparse the signals are
    print(f"\nSignal range: bars {signals['idx'].min()} to {signals['idx'].max()}")
    total_bars = signals['idx'].max() - signals['idx'].min() + 1
    coverage = len(signals) / total_bars * 100
    print(f"Coverage: {len(signals)}/{total_bars} bars ({coverage:.1f}%)")
    
    # Show examples of signal sequences
    print("\n=== Signal Sequences ===")
    for i in range(min(10, len(signals) - 1)):
        curr = signals.iloc[i]
        next_sig = signals.iloc[i + 1]
        gap = next_sig['idx'] - curr['idx']
        if gap > 1:
            print(f"Bar {curr['idx']}: signal={curr['val']} -> Bar {next_sig['idx']}: signal={next_sig['val']} (gap: {gap} bars)")
    
    # The issue: When a stop loss happens at bar X, but there's no signal at bar X
    # the exit memory stores "None" or 0, and then when checking bar X+1, 
    # if there's also no signal, it thinks the signal is unchanged (None == None)
    # and blocks re-entry incorrectly.
    
    print("\n=== The Problem ===")
    print("1. Sparse signal storage means many bars have no signal data")
    print("2. When stop loss occurs at a bar with no signal, exit memory stores None/0")
    print("3. Next bar also has no signal, so None == None, blocking re-entry")
    print("4. But the actual signal might still be active (e.g., -1 for short)")
    
    print("\n=== Solution Options ===")
    print("1. Store the last known signal value in exit memory (not current bar's value)")
    print("2. Disable exit memory when signal data is missing")
    print("3. Forward-fill signal values in the risk manager")
    