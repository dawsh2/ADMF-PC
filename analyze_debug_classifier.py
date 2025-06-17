#!/usr/bin/env python3
"""Analyze classifier trace from debug run."""

import pandas as pd
from pathlib import Path
import json

# Debug workspace
workspace = Path("workspaces/two_layer_debug_test_2d94365f")
classifier_file = workspace / "traces/SPY_1m/classifiers/regime/SPY_market_regime_detector.parquet"

if classifier_file.exists():
    df = pd.read_parquet(classifier_file)
    print(f"Classifier trace exists with {len(df)} changes")
    print(f"First 20 regime changes:")
    for i in range(min(20, len(df))):
        row = df.iloc[i]
        print(f"  {row['ts']} -> {row['val']}")
else:
    print("No classifier trace found!")

# Check metadata
metadata_file = workspace / "metadata.json"
if metadata_file.exists():
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nClassifier component info:")
    if 'components' in metadata:
        for name, info in metadata['components'].items():
            if info.get('component_type') == 'classifier':
                print(f"  {name}: {info['signal_changes']} changes")
                
# Also check if strategy traces exist
strategy_traces = list((workspace / "traces/SPY_1m/signals").rglob("*.parquet"))
print(f"\nFound {len(strategy_traces)} strategy traces")
for trace in strategy_traces[:5]:
    print(f"  {trace.name}")