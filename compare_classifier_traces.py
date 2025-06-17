#!/usr/bin/env python3
"""Compare classifier traces between original run and debug test."""

import pandas as pd
import numpy as np
from pathlib import Path

# Original run
original_workspace = Path("workspaces/two_layer_regime_ensemble_v1_4f71d9e1")
original_classifier = original_workspace / "traces/SPY_1m/classifiers/regime/SPY_market_regime_detector.parquet"

# Debug run  
debug_workspace = Path("workspaces/two_layer_debug_test_2d94365f")
debug_classifier = debug_workspace / "traces/SPY_1m/classifiers/regime/SPY_market_regime_detector.parquet"

print("=== CLASSIFIER TRACE COMPARISON ===\n")

# Analyze original run
if original_classifier.exists():
    df_orig = pd.read_parquet(original_classifier)
    print("ORIGINAL RUN (complete_grid_search):")
    print(f"Total rows: {len(df_orig)}")
    print(f"Unique regimes: {df_orig['val'].unique()}")
    print(f"Regime counts:")
    print(df_orig['val'].value_counts())
    
    # Calculate regime changes
    orig_changes = df_orig[df_orig['val'].shift() != df_orig['val']]
    print(f"\nTotal regime changes: {len(orig_changes)}")
    print(f"First 10 regime changes:")
    for i in range(min(10, len(orig_changes))):
        row = orig_changes.iloc[i]
        print(f"  {row['ts']} -> {row['val']}")
else:
    print("Original classifier trace not found!")

print("\n" + "="*50 + "\n")

# Analyze debug run
if debug_classifier.exists():
    df_debug = pd.read_parquet(debug_classifier)
    print("DEBUG RUN (two_layer_debug):")
    print(f"Total rows: {len(df_debug)}")
    print(f"Unique regimes: {df_debug['val'].unique()}")
    print(f"Regime counts:")
    print(df_debug['val'].value_counts())
    
    # Calculate regime changes
    debug_changes = df_debug[df_debug['val'].shift() != df_debug['val']]
    print(f"\nTotal regime changes: {len(debug_changes)}")
    print(f"All regime changes:")
    for i in range(len(debug_changes)):
        row = debug_changes.iloc[i]
        print(f"  {row['ts']} -> {row['val']}")
else:
    print("Debug classifier trace not found!")

# Check if using same data period
if original_classifier.exists() and debug_classifier.exists():
    print("\n" + "="*50 + "\n")
    print("DATA PERIOD COMPARISON:")
    print(f"Original - First: {df_orig['ts'].iloc[0]}, Last: {df_orig['ts'].iloc[-1]}")
    print(f"Debug    - First: {df_debug['ts'].iloc[0]}, Last: {df_debug['ts'].iloc[-1]}")
    
    # Check for overlap
    orig_start, orig_end = df_orig['ts'].iloc[0], df_orig['ts'].iloc[-1]
    debug_start, debug_end = df_debug['ts'].iloc[0], df_debug['ts'].iloc[-1]
    
    if orig_start <= debug_end and debug_start <= orig_end:
        print("\nData periods overlap!")
        overlap_start = max(orig_start, debug_start)
        overlap_end = min(orig_end, debug_end)
        print(f"Overlap period: {overlap_start} to {overlap_end}")