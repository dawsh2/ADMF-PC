#!/usr/bin/env python3
"""Analyze classifier trace data to understand regime changes."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Workspace path
workspace = Path("workspaces/two_layer_debug_test_2d94365f")

# Load classifier trace
classifier_file = workspace / "traces/SPY_1m/classifiers/regime/SPY_market_regime_detector.parquet"
df_classifier = pd.read_parquet(classifier_file)

print("=== CLASSIFIER TRACE ANALYSIS ===")
print(f"\nTotal rows: {len(df_classifier)}")
print(f"\nColumns: {df_classifier.columns.tolist()}")

# Show first few rows
print("\nFirst 10 rows:")
print(df_classifier.head(10))

# Analyze regime changes
print("\n\n=== REGIME ANALYSIS ===")
print(f"\nUnique regimes: {df_classifier['val'].unique()}")
print(f"\nRegime value counts:")
print(df_classifier['val'].value_counts())

# Calculate regime changes
regime_changes = df_classifier[df_classifier['val'].shift() != df_classifier['val']]
print(f"\nTotal regime changes: {len(regime_changes)}")
print(f"\nRegime change frequency: {len(regime_changes) / len(df_classifier):.2%}")

# Show regime change details
print("\n\n=== REGIME CHANGE DETAILS ===")
print("First 20 regime changes:")
for i in range(min(20, len(regime_changes))):
    row = regime_changes.iloc[i]
    print(f"{i+1}. {row['ts']} - Changed to: {row['val']}")

# Load metadata to understand the run
metadata_file = workspace / "metadata.json"
if metadata_file.exists():
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print("\n\n=== RUN METADATA ===")
    print(f"Run ID: {metadata.get('run_id', 'N/A')}")
    print(f"Start time: {metadata.get('start_time', 'N/A')}")
    print(f"End time: {metadata.get('end_time', 'N/A')}")
    
    if 'strategies' in metadata:
        print(f"\nStrategies configured:")
        for name, config in metadata['strategies'].items():
            print(f"  - {name}: {config}")