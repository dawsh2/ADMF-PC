#!/usr/bin/env python3
"""
Simple analysis of SPY signal generation workspace
"""
import os
import json
from pathlib import Path
from datetime import datetime

def analyze_workspace(workspace_path):
    """Analyze the SPY signal generation workspace"""
    
    workspace = Path(workspace_path)
    if not workspace.exists():
        print(f"Error: Workspace not found at {workspace_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing Workspace: {workspace_path}")
    print(f"{'='*60}\n")
    
    # 1. Check metadata
    metadata_file = workspace / "metadata.json"
    if metadata_file.exists():
        print("1. Workspace Metadata:")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"   - Run ID: {metadata.get('run_id', 'N/A')}")
        print(f"   - Timestamp: {metadata.get('timestamp', 'N/A')}")
        print(f"   - Description: {metadata.get('description', 'N/A')}")
        
        if 'config' in metadata:
            config = metadata['config']
            print(f"\n   Configuration:")
            print(f"   - Symbol: {config.get('symbol', 'N/A')}")
            print(f"   - Start Date: {config.get('start_date', 'N/A')}")
            print(f"   - End Date: {config.get('end_date', 'N/A')}")
        
        if 'strategies' in metadata:
            print(f"\n   - Total Strategies: {len(metadata['strategies'])}")
            
            # Group by type
            strategy_types = {}
            for strategy in metadata['strategies']:
                stype = strategy.get('strategy_type', 'unknown')
                strategy_types[stype] = strategy_types.get(stype, 0) + 1
            
            print(f"   - Strategy Types:")
            for stype, count in sorted(strategy_types.items()):
                print(f"     * {stype}: {count}")
        
        if 'classifiers' in metadata:
            print(f"\n   - Total Classifiers: {len(metadata['classifiers'])}")
    else:
        print("1. No metadata.json found")
    
    # 2. Check traces directory
    traces_dir = workspace / "traces"
    if traces_dir.exists():
        print("\n2. Traces Directory:")
        
        for symbol_dir in traces_dir.iterdir():
            if symbol_dir.is_dir():
                print(f"\n   Symbol: {symbol_dir.name}")
                
                # Check signals
                signals_dir = symbol_dir / "signals"
                if signals_dir.exists():
                    signal_count = 0
                    strategy_dirs = list(signals_dir.iterdir())
                    
                    for strategy_dir in strategy_dirs:
                        if strategy_dir.is_dir():
                            parquet_files = list(strategy_dir.glob("*.parquet"))
                            signal_count += len(parquet_files)
                    
                    print(f"   - Signal Strategies: {len(strategy_dirs)}")
                    print(f"   - Total Signal Files: {signal_count}")
                
                # Check classifiers
                classifiers_dir = symbol_dir / "classifiers"
                if classifiers_dir.exists():
                    classifier_count = 0
                    classifier_dirs = list(classifiers_dir.iterdir())
                    
                    for classifier_dir in classifier_dirs:
                        if classifier_dir.is_dir():
                            parquet_files = list(classifier_dir.glob("*.parquet"))
                            classifier_count += len(parquet_files)
                    
                    print(f"   - Classifier Types: {len(classifier_dirs)}")
                    print(f"   - Total Classifier Files: {classifier_count}")
    else:
        print("\n2. No traces directory found")
    
    # 3. Check for results/metrics
    results_files = list(workspace.glob("**/results*.json"))
    metrics_files = list(workspace.glob("**/metrics*.json"))
    
    if results_files or metrics_files:
        print("\n3. Results/Metrics Files:")
        
        for file in results_files:
            print(f"   - {file.relative_to(workspace)}")
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'performance' in data:
                        print(f"     Performance metrics found")
            except:
                pass
        
        for file in metrics_files:
            print(f"   - {file.relative_to(workspace)}")
    else:
        print("\n3. No results or metrics files found")
    
    # 4. Check for analytics database
    analytics_db = workspace / "analytics.duckdb"
    if analytics_db.exists():
        print(f"\n4. Analytics Database: FOUND ({analytics_db.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("\n4. Analytics Database: NOT FOUND")
    
    # 5. Check for event/signal parquet files at root
    event_files = list(workspace.glob("*/events.parquet"))
    signal_files = list(workspace.glob("*/signals.parquet"))
    
    if event_files or signal_files:
        print("\n5. Container Data Files:")
        print(f"   - Event Files: {len(event_files)}")
        print(f"   - Signal Files: {len(signal_files)}")
        
        # Sample first event file
        if event_files:
            print(f"\n   Sample container: {event_files[0].parent.name}")
    else:
        print("\n5. No container data files found")
    
    # 6. Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    
    # Calculate some basic stats
    total_files = sum(1 for _ in workspace.rglob("*.parquet"))
    total_json = sum(1 for _ in workspace.rglob("*.json"))
    
    print(f"Total Parquet Files: {total_files}")
    print(f"Total JSON Files: {total_json}")
    
    # Check workspace size
    total_size = sum(f.stat().st_size for f in workspace.rglob("*") if f.is_file())
    print(f"Total Workspace Size: {total_size / 1024 / 1024:.1f} MB")
    
    print(f"\n{'='*60}\n")


def main():
    # Target workspace
    workspace_path = "workspaces/20250617_194112_signal_generation_SPY"
    
    # Convert to absolute path
    workspace_abs = Path(workspace_path).absolute()
    
    if not workspace_abs.exists():
        # Try from project root
        project_root = Path(__file__).parent
        workspace_abs = project_root / workspace_path
    
    if workspace_abs.exists():
        analyze_workspace(str(workspace_abs))
    else:
        print(f"Error: Could not find workspace at {workspace_path}")
        print(f"Looked in: {workspace_abs}")
        
        # List available workspaces
        workspaces_dir = Path("workspaces")
        if workspaces_dir.exists():
            print(f"\nAvailable workspaces:")
            for ws in sorted(workspaces_dir.iterdir()):
                if ws.is_dir():
                    print(f"  - {ws.name}")


if __name__ == "__main__":
    main()