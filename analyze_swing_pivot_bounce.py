"""Analyze swing_pivot_bounce workspace results"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Find the most recent swing_pivot_bounce workspace
workspace_path = Path("workspaces/signal_generation_80a6a1d9")

if workspace_path.exists():
    print(f"Analyzing workspace: {workspace_path}")
    
    # Load metadata
    metadata_file = workspace_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"\nTotal bars processed: {metadata.get('total_bars', 0)}")
        print(f"Total signals: {metadata.get('total_signals', 0)}")
        print(f"Stored changes: {metadata.get('stored_changes', 0)}")
        
        # Check components
        components = metadata.get('components', {})
        if components:
            print(f"\nComponents found: {len(components)}")
            for comp_name, comp_data in components.items():
                print(f"  - {comp_name}:")
                print(f"    Type: {comp_data.get('strategy_type', 'unknown')}")
                print(f"    Signal changes: {comp_data.get('signal_changes', 0)}")
                print(f"    Signal file: {comp_data.get('signal_file_path', 'N/A')}")
        else:
            print("\nNo components found in metadata")
    
    # Check for trace files
    traces_dir = workspace_path / "traces"
    if traces_dir.exists():
        print(f"\nChecking traces directory...")
        signal_files = list(traces_dir.rglob("*.parquet"))
        print(f"Found {len(signal_files)} parquet files")
        
        for signal_file in signal_files[:3]:  # Show first 3
            print(f"  - {signal_file.relative_to(workspace_path)}")
            
            # Try to load and check content
            try:
                df = pd.read_parquet(signal_file)
                print(f"    Rows: {len(df)}, Columns: {list(df.columns)}")
                if len(df) > 0:
                    print(f"    First signal: {df.iloc[0].to_dict()}")
            except Exception as e:
                print(f"    Error reading: {e}")
else:
    print(f"Workspace not found: {workspace_path}")

# Also check the feature configuration issue
print("\n=== Feature Configuration Debug ===")
print("The strategy expects features with keys like:")
print("  - support_resistance_20_resistance")
print("  - support_resistance_20_support")
print("\nBut the feature class might be generating different keys.")
print("This mismatch would cause the strategy to return None.")