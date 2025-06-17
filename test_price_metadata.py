#!/usr/bin/env python3
"""
Test script to verify that price metadata is being saved in signal traces.
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import json

def check_price_in_signals(workspace_path: str):
    """Check if price data is being saved in signal traces."""
    workspace = Path(workspace_path)
    traces_dir = workspace / "traces"
    
    if not traces_dir.exists():
        print(f"No traces directory found at {traces_dir}")
        return
    
    # Find all signal parquet files
    signal_files = list(traces_dir.rglob("signals/*/*.parquet"))
    
    if not signal_files:
        print("No signal files found")
        return
    
    print(f"Found {len(signal_files)} signal files")
    
    for signal_file in signal_files[:5]:  # Check first 5 files
        print(f"\nChecking: {signal_file.relative_to(workspace)}")
        
        # Read parquet file
        df = pd.read_parquet(signal_file)
        
        # Check if price column exists
        if 'px' in df.columns:
            print(f"  ✓ Price column found")
            # Show some sample prices
            non_zero_prices = df[df['px'] != 0.0]
            if len(non_zero_prices) > 0:
                print(f"  Sample prices: {non_zero_prices['px'].head().tolist()}")
                print(f"  Price range: ${non_zero_prices['px'].min():.2f} - ${non_zero_prices['px'].max():.2f}")
            else:
                print("  ⚠️  All prices are 0.0")
        else:
            print(f"  ✗ No price column found")
            print(f"  Available columns: {df.columns.tolist()}")
        
        # Check metadata if available
        try:
            parquet_file = pq.ParquetFile(signal_file)
            metadata = parquet_file.metadata.metadata
            if metadata and b'strategy_parameters' in metadata:
                params = json.loads(metadata[b'strategy_parameters'].decode())
                print(f"  Strategy params: {list(params.keys())}")
        except Exception as e:
            print(f"  Could not read metadata: {e}")

if __name__ == "__main__":
    # Replace with your actual workspace path
    workspace_path = "./workspaces/duckdb_ensemble_cost_optimized_v1_38cfe9a0"
    check_price_in_signals(workspace_path)