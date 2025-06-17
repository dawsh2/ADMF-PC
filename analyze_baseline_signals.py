#!/usr/bin/env python3
"""Analyze why baseline strategies were silent."""

import pandas as pd
from pathlib import Path

# Test workspace
workspace_path = Path("./workspaces/two_layer_regime_ensemble_v1_3594c2a1")

# Load one of the ensemble signal files
signal_file = workspace_path / "traces/SPY_1m/signals/ma_crossover/SPY_baseline_plus_regime_boosters_{'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 15}}.parquet"

print("=== ANALYZING BASELINE SILENCE ===\n")

if signal_file.exists():
    df_signals = pd.read_parquet(signal_file)
    print(f"Total signal changes: {len(df_signals)}")
    print(f"Signal file columns: {df_signals.columns.tolist()}\n")
    
    # Show first 20 signal changes
    print("First 20 signal changes:")
    for i in range(min(20, len(df_signals))):
        row = df_signals.iloc[i]
        print(f"  Bar {row.get('idx', 'N/A')}: Signal = {row.get('val', 'N/A')}, Price = ${row.get('px', 0):.2f}")
    
    # Analyze signal patterns
    print(f"\nSignal value distribution:")
    print(df_signals['val'].value_counts())
    
    # Check for metadata if available
    if 'metadata' in df_signals.columns:
        print("\nChecking metadata for strategy details...")
        # Sample a few rows with metadata
        for i in range(min(5, len(df_signals))):
            if pd.notna(df_signals.iloc[i].get('metadata')):
                print(f"\nBar {df_signals.iloc[i]['idx']} metadata:")
                print(df_signals.iloc[i]['metadata'])
else:
    print(f"Signal file not found: {signal_file}")

# Also check the classifier
classifier_file = workspace_path / "traces/SPY_1m/classifiers/regime/SPY_market_regime_detector.parquet"
if classifier_file.exists():
    df_classifier = pd.read_parquet(classifier_file)
    print(f"\n\nClassifier regime distribution:")
    print(df_classifier['val'].value_counts())
    
# Let's also check if we can find individual baseline strategy traces
# The baseline strategies should have been run separately too
print("\n\nLooking for individual strategy traces...")
for subdir in ['ma_crossover', 'regime', 'momentum', 'oscillator']:
    trace_dir = workspace_path / f"traces/SPY_1m/signals/{subdir}"
    if trace_dir.exists():
        traces = list(trace_dir.glob("*.parquet"))
        if traces:
            print(f"\n{subdir} traces found: {len(traces)}")
            for trace in traces[:3]:  # Show first 3
                print(f"  - {trace.name}")