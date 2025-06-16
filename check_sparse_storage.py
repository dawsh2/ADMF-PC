#!/usr/bin/env python3
"""Check sparse storage format without pandas."""

import json
from pathlib import Path

# Let's first check what's in the traces
traces_dir = Path("/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_3fabc3f9/traces/SPY_1m")

print("=== TRACE DIRECTORY STRUCTURE ===")
if traces_dir.exists():
    for category in ['signals', 'classifiers']:
        cat_dir = traces_dir / category
        if cat_dir.exists():
            print(f"\n{category.upper()}:")
            for strategy_dir in sorted(cat_dir.iterdir()):
                if strategy_dir.is_dir():
                    files = list(strategy_dir.glob("*.parquet"))
                    print(f"  {strategy_dir.name}: {len(files)} files")
                    if files and len(files) > 0:
                        print(f"    Example: {files[0].name}")

# Check if there are any CSV versions we can examine
print("\n=== LOOKING FOR CSV FILES ===")
csv_files = list(traces_dir.rglob("*.csv"))
if csv_files:
    print(f"Found {len(csv_files)} CSV files")
    for f in csv_files[:5]:
        print(f"  {f.relative_to(traces_dir)}")
else:
    print("No CSV files found")

# Check for any metadata files
print("\n=== LOOKING FOR METADATA ===")
meta_files = list(traces_dir.rglob("*meta*"))
if meta_files:
    for f in meta_files[:5]:
        print(f"  {f.relative_to(traces_dir)}")
        
# Let's see what's in the workspace root
workspace_root = Path("/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_3fabc3f9")
print(f"\n=== WORKSPACE ROOT CONTENTS ===")
for item in sorted(workspace_root.iterdir()):
    if item.is_file():
        print(f"  File: {item.name}")
    else:
        print(f"  Dir: {item.name}/")