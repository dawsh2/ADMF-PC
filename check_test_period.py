#!/usr/bin/env python3
"""Check the test period and warmup requirements."""

import pandas as pd
from pathlib import Path

# Debug workspace (short run)
debug_workspace = Path("./workspaces/two_layer_debug_test_2d94365f")

# Full test workspace  
test_workspace = Path("./workspaces/two_layer_regime_ensemble_v1_3594c2a1")

print("=== WARMUP ANALYSIS ===\n")

# Check debug run signals
debug_signal = debug_workspace / "traces/SPY_1m/signals/ma_crossover/SPY_debug_ensemble_{'name': 'sma_crossover', 'params': {'fast_period': 19, 'slow_period': 15}}.parquet"
if debug_signal.exists():
    df_debug = pd.read_parquet(debug_signal)
    print(f"DEBUG RUN (20,000 bars):")
    print(f"  Total signal changes: {len(df_debug)}")
    if len(df_debug) > 0:
        print(f"  First signal at bar: {df_debug.iloc[0]['idx']}")
        print(f"  Last signal at bar: {df_debug.iloc[-1]['idx']}")
    else:
        print("  NO SIGNALS GENERATED!")

# Check test run signals
test_signal = test_workspace / "traces/SPY_1m/signals/ma_crossover/SPY_baseline_plus_regime_boosters_{'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 15}}.parquet"
if test_signal.exists():
    df_test = pd.read_parquet(test_signal)
    print(f"\nFULL TEST RUN (102,235 bars):")
    print(f"  Total signal changes: {len(df_test)}")
    if len(df_test) > 0:
        print(f"  First signal at bar: {df_test.iloc[0]['idx']}")
        print(f"  Last signal at bar: {df_test.iloc[-1]['idx']}")
        print(f"  Active trading bars: {df_test.iloc[-1]['idx'] - df_test.iloc[0]['idx'] + 1}")
        print(f"  Warmup period: {df_test.iloc[0]['idx']} bars")

# Calculate what indicators might need this warmup
print("\n\nPOSSIBLE WARMUP REQUIREMENTS:")
print("From config, the strategies use:")
print("  - ichimoku_cloud_position: conversion=9, base=35")
print("  - williams_r: period=21")  
print("  - ema_sma_crossover: ema=5, sma=50")
print("  - aroon_crossover: period=14")
print("  - pivot_channel_bounces: sr_period=15")
print("  - And various others...")

print("\nThe 81,812 bar warmup suggests something needs ~81,812 bars of history.")
print("This is way more than any individual indicator period!")

# Let's check the metadata to see if there's more info
metadata_file = test_workspace / "metadata.json"
if metadata_file.exists():
    import json
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Check strategy metadata for clues
    if 'strategy_metadata' in metadata and 'strategies' in metadata['strategy_metadata']:
        first_strategy = list(metadata['strategy_metadata']['strategies'].values())[0]
        if 'params' in first_strategy:
            print("\nChecking ensemble parameters...")
            params = first_strategy['params']
            if 'baseline_strategies' in params:
                print("Baseline strategies in ensemble:")
                baseline = params['baseline_strategies']
                if isinstance(baseline, list):
                    for s in baseline:
                        print(f"  - {s['name']}: {s['params']}")
                else:
                    print(f"  - {baseline['name']}: {baseline['params']}")