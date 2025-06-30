#!/usr/bin/env python3
"""Debug signal tracking and exit memory."""

import pandas as pd
import json
from pathlib import Path

print("=== Debugging Signal Tracking ===")

results_dir = Path("config/bollinger/results/latest")

# Load all data
pos_close_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"
signals_file = results_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"

if all(f.exists() for f in [pos_close_file, signals_file]):
    closes = pd.read_parquet(pos_close_file)
    signals = pd.read_parquet(signals_file)
    
    # Parse metadata
    if 'metadata' in closes.columns:
        for i in range(len(closes)):
            if isinstance(closes.iloc[i]['metadata'], str):
                try:
                    meta = json.loads(closes.iloc[i]['metadata'])
                    for key, value in meta.items():
                        if key not in closes.columns:
                            closes.loc[closes.index[i], key] = value
                except:
                    pass
    
    print(f"Total position closes: {len(closes)}")
    print(f"Total signals: {len(signals)}")
    
    # Check risk exits
    risk_exits = closes[closes['exit_type'].isin(['stop_loss', 'take_profit'])]
    print(f"\nRisk exits: {len(risk_exits)}")
    
    # For each risk exit, check what signal was active
    print("\n=== First 10 Risk Exits ===")
    for idx, (i, exit) in enumerate(risk_exits.head(10).iterrows()):
        exit_bar = exit['idx']
        
        # Find the last signal before or at exit
        signals_before = signals[signals['idx'] <= exit_bar].sort_values('idx')
        if len(signals_before) > 0:
            last_signal = signals_before.iloc[-1]
            bars_since_signal = exit_bar - last_signal['idx']
            
            print(f"\n{idx+1}. {exit['exit_type']} at bar {exit_bar}")
            print(f"   Last signal: {last_signal['val']} at bar {last_signal['idx']} ({bars_since_signal} bars before exit)")
            
            # What signal should be stored in exit memory?
            print(f"   Exit memory should store: {last_signal['val']}")
            
            # Check next few signals
            next_signals = signals[signals['idx'] > exit_bar].head(3)
            if len(next_signals) > 0:
                print("   Next signals:")
                for _, sig in next_signals.iterrows():
                    print(f"     Bar {sig['idx']}: {sig['val']}")
    
    # Check signal frequency
    print("\n=== Signal Frequency ===")
    signal_gaps = signals['idx'].diff().dropna()
    print(f"Average bars between signals: {signal_gaps.mean():.1f}")
    print(f"Max gap: {signal_gaps.max()}")
    
    # The issue might be sparse signals
    print("\n=== Signal Coverage ===")
    total_bars = signals['idx'].max() - signals['idx'].min() + 1
    signal_coverage = len(signals) / total_bars * 100
    print(f"Signal coverage: {len(signals)}/{total_bars} bars ({signal_coverage:.1f}%)")
    
    print("\n=== Potential Issues ===")
    print("1. With sparse signals (20% coverage), exit memory might not have current signal")
    print("2. Portfolio should track last signal value per strategy")
    print("3. Risk manager should use that tracked value, not current bar's signal")