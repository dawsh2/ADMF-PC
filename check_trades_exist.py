#!/usr/bin/env python3
"""Check if trades file exists and where."""

from pathlib import Path
import os

print("=== Checking for Trades Files ===")

# Check various possible locations
locations = [
    "config/bollinger/results/latest/traces/events/portfolio/trades.parquet",
    "config/bollinger/results/latest/traces/trades.parquet",
    "config/bollinger/results/latest/trades.parquet",
    "results/latest/traces/events/portfolio/trades.parquet",
    "results/latest/trades.parquet"
]

found = False
for loc in locations:
    path = Path(loc)
    if path.exists():
        print(f"✓ Found trades at: {loc}")
        found = True
        
        # Check file size
        size = os.path.getsize(path)
        print(f"  File size: {size:,} bytes")
        
        # Try to load it
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            print(f"  Number of trades: {len(df)}")
            print(f"  Columns: {list(df.columns)[:5]}...")  # First 5 columns
        except Exception as e:
            print(f"  Error loading: {e}")

if not found:
    print("\n❌ No trades file found!")
    
    # Check if results directory exists
    results_dir = Path("config/bollinger/results")
    if results_dir.exists():
        print(f"\nResults directory exists: {results_dir}")
        
        # List subdirectories
        subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
        if subdirs:
            print(f"Found {len(subdirs)} result directories:")
            for d in sorted(subdirs)[-5:]:  # Last 5
                print(f"  - {d.name}")
        
        # Check latest symlink
        latest = results_dir / "latest"
        if latest.exists():
            print(f"\nLatest symlink points to: {latest.resolve()}")
    else:
        print(f"\nResults directory does not exist: {results_dir}")

print("\n\nTo see your 463 trades, you need to:")
print("1. Make sure you ran the backtest in this directory")
print("2. Check that it completed successfully")
print("3. Look for the trades file in the correct location")