#!/usr/bin/env python3
"""
Analyze signal generation results from traces
"""
import os
import json
import pandas as pd
from collections import defaultdict

workspace = "workspaces/expansive_grid_search_5fe966d1"

# Load metadata to get all configured strategies
with open(f"{workspace}/metadata.json", "r") as f:
    metadata = json.load(f)

total_strategies = len(metadata['strategies'])
print(f"Total strategies configured: {total_strategies}")

# Find all signal parquet files
signal_files = []
signals_dir = f"{workspace}/traces/SPY_1m/signals"
for strategy_dir in os.listdir(signals_dir):
    strategy_path = os.path.join(signals_dir, strategy_dir)
    if os.path.isdir(strategy_path):
        for file in os.listdir(strategy_path):
            if file.endswith('.parquet'):
                signal_files.append(os.path.join(strategy_path, file))

print(f"\nSignal files generated: {len(signal_files)}")

# Group by strategy type
strategy_counts = defaultdict(int)
strategy_with_signals = set()

for file_path in signal_files:
    # Extract strategy info from path
    parts = file_path.split('/')
    strategy_type = parts[-2].replace('_grid', '')
    strategy_id = parts[-1].replace('.parquet', '')
    strategy_counts[strategy_type] += 1
    strategy_with_signals.add(strategy_id)

print(f"\nStrategies that generated signals: {len(strategy_with_signals)}")

print("\nSignals by strategy type:")
for strategy_type, count in sorted(strategy_counts.items()):
    print(f"  {strategy_type}: {count} parameter combinations")

# Find strategies without signals
all_strategy_ids = {s['strategy_id'] for s in metadata['strategies']}
strategies_without_signals = all_strategy_ids - strategy_with_signals

print(f"\nStrategies without signals: {len(strategies_without_signals)}")

# Group strategies without signals by type
no_signal_by_type = defaultdict(list)
for strat_id in strategies_without_signals:
    strategy_type = strat_id.split('_')[0]
    no_signal_by_type[strategy_type].append(strat_id)

print("\nStrategy types that generated NO signals:")
for strategy_type, strat_list in sorted(no_signal_by_type.items()):
    if len(strat_list) > 10:  # Type that completely failed
        print(f"  {strategy_type}: {len(strat_list)} parameter combinations (ALL FAILED)")
    else:
        print(f"  {strategy_type}: {len(strat_list)} parameter combinations")

# Check a sample signal file to see signal distribution
if signal_files:
    print("\n=== SAMPLE SIGNAL ANALYSIS ===")
    sample_file = signal_files[0]
    print(f"Analyzing: {sample_file}")
    
    try:
        df = pd.read_parquet(sample_file)
        print(f"Total signals in file: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Signal value distribution
        print("\nSignal value distribution:")
        signal_dist = df['signal_value'].value_counts().sort_index()
        for value, count in signal_dist.items():
            print(f"  {value}: {count} ({count/len(df)*100:.1f}%)")
    except Exception as e:
        print(f"Error reading parquet file: {e}")

# Check classifier results
classifiers_dir = f"{workspace}/traces/SPY_1m/classifiers"
if os.path.exists(classifiers_dir):
    classifier_files = []
    for classifier_dir in os.listdir(classifiers_dir):
        classifier_path = os.path.join(classifiers_dir, classifier_dir)
        if os.path.isdir(classifier_path):
            for file in os.listdir(classifier_path):
                if file.endswith('.parquet'):
                    classifier_files.append(os.path.join(classifier_path, file))
    
    print(f"\n=== CLASSIFIER RESULTS ===")
    print(f"Classifier files generated: {len(classifier_files)}")
    
    # Check metadata for expected classifiers
    if 'classifiers' in metadata:
        print(f"Total classifiers configured: {len(metadata['classifiers'])}")