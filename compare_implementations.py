#!/usr/bin/env python3
"""
Compare how signals are processed in parameter sweep vs ensemble.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_signal_persistence(signals_df, label):
    """Analyze how signals persist in the sparse format."""
    print(f"\n{label} Signal Persistence Analysis:")
    print("-" * 50)
    
    # Look at signal transitions
    transitions = []
    for i in range(1, len(signals_df)):
        prev_val = signals_df.iloc[i-1]['val']
        curr_val = signals_df.iloc[i]['val']
        bar_gap = signals_df.iloc[i]['idx'] - signals_df.iloc[i-1]['idx']
        
        transitions.append({
            'from': prev_val,
            'to': curr_val,
            'bar_gap': bar_gap,
            'type': 'exit' if curr_val == 0 else 'entry' if prev_val == 0 else 'reversal'
        })
    
    df_trans = pd.DataFrame(transitions)
    
    # Count transition types
    print("\nTransition types:")
    print(df_trans['type'].value_counts())
    
    # Average gaps between signals
    print(f"\nAverage bars between signal changes: {df_trans['bar_gap'].mean():.1f}")
    
    # Look for immediate exits (signal followed by 0 on next bar)
    immediate_exits = df_trans[(df_trans['type'] == 'exit') & (df_trans['bar_gap'] == 1)]
    print(f"\nImmediate exits (1 bar): {len(immediate_exits)} ({len(immediate_exits)/len(df_trans)*100:.1f}%)")
    
    # Pattern analysis
    print("\nSignal patterns (first 20 transitions):")
    for i, row in df_trans.head(20).iterrows():
        print(f"  {row['from']} -> {row['to']} (gap: {row['bar_gap']} bars, {row['type']})")

def main():
    print("Comparing Signal Processing: Parameter Sweep vs Ensemble")
    print("=" * 60)
    
    # Load both signal files
    # Strategy 40 = (15-10)*7 + 5 = 35+5 = 40
    sweep_path = Path("config/bollinger/results/latest/traces/bollinger_bands/SPY_5m_compiled_strategy_40.parquet")
    ensemble_path = Path("config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet")
    
    if sweep_path.exists():
        sweep_signals = pd.read_parquet(sweep_path)
        analyze_signal_persistence(sweep_signals, "Parameter Sweep")
    else:
        print(f"Sweep file not found: {sweep_path}")
    
    if ensemble_path.exists():
        ensemble_signals = pd.read_parquet(ensemble_path)
        analyze_signal_persistence(ensemble_signals, "Ensemble")
    else:
        print(f"Ensemble file not found: {ensemble_path}")
    
    # Key insight
    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("If the ensemble has many immediate exits (1-bar holdings), it suggests")
    print("the exit condition is being evaluated differently than in the sweep.")
    print("\nPossible causes:")
    print("1. Feature calculation differences")
    print("2. Signal sparse storage/expansion differences")
    print("3. Exit condition evaluation timing")

if __name__ == "__main__":
    main()