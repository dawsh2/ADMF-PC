#!/usr/bin/env python3
"""Analyze why we still have 53 immediate re-entries."""

import pandas as pd
import json
from pathlib import Path

results_dir = Path("config/bollinger/results/latest")

# Load all data
opens = pd.read_parquet(results_dir / "traces/portfolio/positions_open/positions_open.parquet")
closes = pd.read_parquet(results_dir / "traces/portfolio/positions_close/positions_close.parquet")
signals = pd.read_parquet(results_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet")

# Parse metadata
for df in [opens, closes]:
    if 'metadata' in df.columns:
        for i in range(len(df)):
            if isinstance(df.iloc[i]['metadata'], str):
                try:
                    meta = json.loads(df.iloc[i]['metadata'])
                    # Handle nested metadata
                    if 'metadata' in meta:
                        inner_meta = meta['metadata']
                        for key, value in inner_meta.items():
                            df.loc[df.index[i], key] = value
                    # Also extract top-level fields
                    for key in ['entry_signal', 'exit_type', 'exit_reason']:
                        if key in meta:
                            df.loc[df.index[i], key] = meta[key]
                except:
                    pass

print("=== Analyzing Remaining 53 Immediate Re-entries ===\n")

# Find immediate re-entries
immediate_reentries = []
for i in range(min(len(opens), len(closes)) - 1):
    close_event = closes.iloc[i]
    exit_type = close_event.get('exit_type', 'unknown')
    
    if exit_type in ['stop_loss', 'take_profit', 'trailing_stop']:
        next_open = opens.iloc[i + 1]
        bars_between = next_open['idx'] - close_event['idx']
        
        if bars_between <= 1:
            # Get signal at close and reopen
            signal_at_close = signals[signals['idx'] <= close_event['idx']].iloc[-1]['val'] if len(signals[signals['idx'] <= close_event['idx']]) > 0 else None
            signal_at_open = signals[signals['idx'] <= next_open['idx']].iloc[-1]['val'] if len(signals[signals['idx'] <= next_open['idx']]) > 0 else None
            
            immediate_reentries.append({
                'trade_num': i,
                'close_bar': close_event['idx'],
                'exit_type': exit_type,
                'open_bar': next_open['idx'],
                'bars_between': bars_between,
                'entry_signal_stored': close_event.get('entry_signal', 'NOT_FOUND'),
                'signal_at_close': signal_at_close,
                'signal_at_open': signal_at_open,
                'signal_changed': signal_at_close != signal_at_open
            })

print(f"Total immediate re-entries: {len(immediate_reentries)}")

# Analyze patterns
print("\n=== Pattern Analysis ===")

# Check if entry_signal was stored
no_entry_signal = [r for r in immediate_reentries if r['entry_signal_stored'] == 'NOT_FOUND']
print(f"\nRe-entries without entry_signal stored: {len(no_entry_signal)}")

# Check signal changes
signal_unchanged = [r for r in immediate_reentries if not r['signal_changed']]
print(f"Re-entries where signal didn't change: {len(signal_unchanged)}")

# Group by specific patterns
print("\n=== Specific Patterns ===")

# Pattern 1: Exit at FLAT signal
exit_at_flat = []
for r in immediate_reentries:
    if r['signal_at_close'] == 0:  # FLAT
        exit_at_flat.append(r)

print(f"\n1. Exits at FLAT signal: {len(exit_at_flat)}")
if exit_at_flat:
    print("   These might be legitimate - we exit at FLAT and re-enter when signal becomes directional")
    print(f"   First example: Close bar {exit_at_flat[0]['close_bar']}, signal {exit_at_flat[0]['signal_at_close']} -> {exit_at_flat[0]['signal_at_open']}")

# Pattern 2: Same signal re-entry
same_signal = []
for r in immediate_reentries:
    if r['signal_at_close'] == r['signal_at_open'] and r['signal_at_close'] != 0:
        same_signal.append(r)

print(f"\n2. Same signal re-entries (exit memory should block): {len(same_signal)}")
if same_signal:
    for i, r in enumerate(same_signal[:3]):
        print(f"   Example {i+1}: Trade #{r['trade_num']}, signal={r['signal_at_close']}, exit_type={r['exit_type']}")

# Pattern 3: Signal reversal
reversals = []
for r in immediate_reentries:
    if r['signal_at_close'] * r['signal_at_open'] < 0:  # Different signs
        reversals.append(r)

print(f"\n3. Signal reversals (legitimate): {len(reversals)}")

print("\n=== Summary ===")
print(f"Exit memory should have blocked: {len(same_signal)} trades")
print(f"Legitimate re-entries: {len(immediate_reentries) - len(same_signal)} trades")