#!/usr/bin/env python3
"""
Quick test to see which strategies are generating signals.
Runs grid search for minimal time and tracks results.
"""

import os
import sys
import time
import subprocess
import sqlite3
import pandas as pd
from pathlib import Path
import yaml


def run_grid_search_briefly():
    """Run grid search for a very short time"""
    print("Starting grid search (5 seconds only)...")
    
    # Start the grid search process
    process = subprocess.Popen(
        [sys.executable, "src/analytics/cli/grid_search.py", "expansive_grid_search"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Let it run for 5 seconds
    time.sleep(5)
    
    # Terminate the process
    process.terminate()
    process.wait()
    
    print("Grid search terminated after 5 seconds")
    
    # Find the most recent workspace
    workspaces = list(Path("workspaces").glob("expansive_grid_search_*"))
    if not workspaces:
        print("No workspace created")
        return None
    
    latest = max(workspaces, key=lambda p: p.stat().st_mtime)
    print(f"Checking workspace: {latest}")
    
    return latest


def analyze_quick_results(workspace_path):
    """Analyze results from quick run"""
    # Check for database file
    db_path = workspace_path / "results.db"
    duckdb_path = workspace_path / "analytics.duckdb"
    
    if db_path.exists():
        print(f"\nFound SQLite database: {db_path}")
        conn = sqlite3.connect(db_path)
        
        # Get strategy results
        try:
            df = pd.read_sql("SELECT type, COUNT(*) as count, SUM(total_signals) as signals FROM strategies GROUP BY type", conn)
            print("\nStrategy results:")
            print(df)
        except Exception as e:
            print(f"Error reading strategies: {e}")
        
        conn.close()
    
    elif duckdb_path.exists():
        print(f"\nFound DuckDB database: {duckdb_path}")
        # Would need duckdb module to read this
    
    # Check metadata
    metadata_path = workspace_path / "metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        if 'strategies' in metadata:
            print(f"\nTotal strategies in metadata: {len(metadata['strategies'])}")
            
            # Group by type
            by_type = {}
            for s in metadata['strategies']:
                stype = s.get('type', 'unknown')
                if stype not in by_type:
                    by_type[stype] = 0
                by_type[stype] += 1
            
            print("\nStrategies by type:")
            for stype, count in sorted(by_type.items()):
                print(f"  {stype}: {count}")


def check_existing_workspaces():
    """Check most recent workspaces for results"""
    print("\nChecking existing workspaces...")
    
    # Find recent workspaces
    workspaces = []
    for pattern in ["expansive_grid_search_*", "indicator_grid_v3_*", "20250615_*"]:
        workspaces.extend(Path("workspaces").glob(pattern))
    
    if not workspaces:
        print("No workspaces found")
        return
    
    # Sort by modification time
    workspaces.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Check the most recent ones
    for i, workspace in enumerate(workspaces[:3]):
        print(f"\n{i+1}. {workspace.name}")
        analyze_quick_results(workspace)
        
        if i == 0:  # Most recent
            return workspace


def main():
    """Main test"""
    print("=" * 80)
    print("QUICK STRATEGY TEST")
    print("=" * 80)
    
    # First check existing workspaces
    latest = check_existing_workspaces()
    
    # Run new test if no recent results
    if not latest or input("\nRun new test? (y/n): ").lower() == 'y':
        workspace = run_grid_search_briefly()
        if workspace:
            analyze_quick_results(workspace)


if __name__ == "__main__":
    main()