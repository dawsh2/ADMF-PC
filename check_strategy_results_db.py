#!/usr/bin/env python3
"""
Check strategy results directly from the database.
Analyzes which strategies are generating signals in the most recent run.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from collections import defaultdict
import yaml


def get_latest_workspace():
    """Find the most recent workspace"""
    workspaces_dir = Path("workspaces")
    
    # Find all grid search workspaces (both naming patterns)
    grid_workspaces = [d for d in workspaces_dir.iterdir() 
                      if d.is_dir() and (d.name.startswith("expansive_grid_search_") or 
                                        d.name.startswith("indicator_grid_v3_"))]
    
    if not grid_workspaces:
        print("No expansive_grid_search workspaces found")
        return None
    
    # Sort by modification time
    latest = max(grid_workspaces, key=lambda d: d.stat().st_mtime)
    return latest


def analyze_workspace_results(workspace_path):
    """Analyze results from a workspace"""
    db_path = workspace_path / "results.db"
    
    if not db_path.exists():
        print(f"No results.db found in {workspace_path}")
        return None
    
    conn = sqlite3.connect(db_path)
    
    # Get all tables
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f"\nTables in database: {', '.join(tables['name'].tolist())}")
    
    # Check strategies table
    if 'strategies' in tables['name'].values:
        strategies_df = pd.read_sql("SELECT * FROM strategies", conn)
        print(f"\nTotal strategies in database: {len(strategies_df)}")
        
        # Group by type
        by_type = strategies_df.groupby('type').agg({
            'strategy_id': 'count',
            'total_signals': 'sum'
        }).rename(columns={'strategy_id': 'count'})
        
        print("\nStrategies by type:")
        print(by_type.sort_values('total_signals', ascending=False))
        
        # Find strategies with no signals
        no_signals = strategies_df[strategies_df['total_signals'] == 0]
        print(f"\nStrategies with NO signals: {len(no_signals)}")
        
        if len(no_signals) > 0:
            no_signal_types = no_signals.groupby('type').size()
            print("\nStrategy types with no signals:")
            for stype, count in no_signal_types.items():
                print(f"  - {stype}: {count} instances")
    
    # Check signals table
    if 'signals' in tables['name'].values:
        try:
            signal_count = pd.read_sql("SELECT COUNT(*) as count FROM signals", conn).iloc[0]['count']
            print(f"\nTotal signals in database: {signal_count}")
            
            # Get signal distribution by strategy type
            signals_by_type = pd.read_sql("""
                SELECT strategy_type, COUNT(*) as signal_count
                FROM signals
                GROUP BY strategy_type
                ORDER BY signal_count DESC
            """, conn)
            
            print("\nSignals by strategy type:")
            for _, row in signals_by_type.iterrows():
                print(f"  - {row['strategy_type']}: {row['signal_count']} signals")
        except Exception as e:
            print(f"Error reading signals table: {e}")
    
    conn.close()
    
    return strategies_df if 'strategies' in tables['name'].values else None


def compare_with_config():
    """Compare database results with expected config"""
    # Load config
    config_path = "config/expansive_grid_search.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get topology config
    topology_config = config.get('topology', {})
    if 'file' in topology_config:
        topology_path = Path(topology_config['file'])
        if not topology_path.is_absolute():
            topology_path = Path(config_path).parent / topology_path
        
        with open(topology_path, 'r') as f:
            topology_data = yaml.safe_load(f)
            topology_config = topology_data.get('topology', topology_config)
    
    # Extract expected strategy types
    expected_types = set()
    if 'strategies' in topology_config:
        for strategy_def in topology_config['strategies']:
            if isinstance(strategy_def, dict) and 'type' in strategy_def:
                expected_types.add(strategy_def['type'])
    
    print(f"\nExpected strategy types from config: {len(expected_types)}")
    for stype in sorted(expected_types):
        print(f"  - {stype}")
    
    return expected_types


def main():
    """Main analysis"""
    print("=" * 80)
    print("STRATEGY RESULTS DATABASE ANALYSIS")
    print("=" * 80)
    
    # Get latest workspace
    workspace = get_latest_workspace()
    if not workspace:
        return
    
    print(f"\nAnalyzing workspace: {workspace.name}")
    print(f"Path: {workspace}")
    
    # Analyze results
    strategies_df = analyze_workspace_results(workspace)
    
    # Compare with config
    print("\n" + "=" * 80)
    print("COMPARISON WITH CONFIG:")
    print("=" * 80)
    
    expected_types = compare_with_config()
    
    if strategies_df is not None:
        actual_types = set(strategies_df['type'].unique())
        
        print(f"\nActual strategy types in database: {len(actual_types)}")
        
        # Find differences
        missing_types = expected_types - actual_types
        extra_types = actual_types - expected_types
        
        if missing_types:
            print(f"\nMissing strategy types ({len(missing_types)}):")
            for stype in sorted(missing_types):
                print(f"  ✗ {stype}")
        
        if extra_types:
            print(f"\nExtra strategy types ({len(extra_types)}):")
            for stype in sorted(extra_types):
                print(f"  ? {stype}")
        
        # Working vs not working
        working_df = strategies_df[strategies_df['total_signals'] > 0]
        not_working_df = strategies_df[strategies_df['total_signals'] == 0]
        
        working_types = set(working_df['type'].unique())
        not_working_types = set(not_working_df['type'].unique()) - working_types
        
        print(f"\n" + "=" * 80)
        print("FINAL SUMMARY:")
        print("=" * 80)
        print(f"Expected types: {len(expected_types)}")
        print(f"Actual types: {len(actual_types)}")
        print(f"Working types: {len(working_types)}")
        print(f"Not working types: {len(not_working_types)}")
        print(f"Missing types: {len(missing_types)}")
        
        print("\n" + "=" * 80)
        print("NOT WORKING STRATEGY TYPES:")
        print("=" * 80)
        for stype in sorted(not_working_types):
            count = len(not_working_df[not_working_df['type'] == stype])
            print(f"  ✗ {stype} ({count} instances)")
        
        # Save summary
        summary = {
            'workspace': workspace.name,
            'expected_types': sorted(expected_types),
            'working_types': sorted(working_types),
            'not_working_types': sorted(not_working_types),
            'missing_types': sorted(missing_types)
        }
        
        with open('strategy_check_summary.json', 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to strategy_check_summary.json")


if __name__ == "__main__":
    main()