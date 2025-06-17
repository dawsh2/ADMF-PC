#!/usr/bin/env python3
"""
Debug ensemble signal indices to understand the mismatch.
"""

import pandas as pd
from pathlib import Path

WORKSPACE_PATH = "/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_c6dcf7c0"
DATA_PATH = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"

# Load data to get total bars
data = pd.read_parquet(DATA_PATH)
total_bars = len(data)
print(f"Total bars in SPY data: {total_bars:,}")

# Load signal file
signal_file = Path(WORKSPACE_PATH) / "traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet"
signals_df = pd.read_parquet(signal_file)

print(f"\nSignal file analysis:")
print(f"Total signals: {len(signals_df)}")
print(f"Index range: {signals_df['idx'].min()} to {signals_df['idx'].max()}")
print(f"First 10 indices: {signals_df['idx'].head(10).tolist()}")
print(f"Last 10 indices: {signals_df['idx'].tail(10).tolist()}")

# Check what we're looking for
analysis_start = total_bars - 22000
print(f"\nAnalysis window:")
print(f"Looking for indices >= {analysis_start:,}")
print(f"But max signal index is: {signals_df['idx'].max():,}")
print(f"Gap: {analysis_start - signals_df['idx'].max():,} bars")

# Check metadata
metadata_file = Path(WORKSPACE_PATH) / "metadata.json"
if metadata_file.exists():
    import json
    with open(metadata_file) as f:
        metadata = json.load(f)
    print(f"\nWorkspace metadata:")
    print(f"Total bars processed: {metadata.get('total_bars', 'N/A')}")
    print(f"Workspace path: {metadata.get('workspace_path', 'N/A')}")