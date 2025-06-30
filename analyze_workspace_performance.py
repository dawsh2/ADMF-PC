#!/usr/bin/env python3
"""
Analyze P&L and performance metrics from ADMF-PC workspace
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add src to Python path for imports
sys.path.append(str(Path(__file__).parent))

from src.analytics.workspace import AnalyticsWorkspace
from src.analytics.signal_performance_analyzer import SignalPerformanceAnalyzer


def analyze_workspace_signals(workspace_path: str):
    """Analyze signals and calculate P&L from a workspace"""
    
    print(f"\n{'='*60}")
    print(f"Analyzing Workspace: {workspace_path}")
    print(f"{'='*60}\n")
    
    workspace = Path(workspace_path)
    if not workspace.exists():
        print(f"Error: Workspace not found at {workspace_path}")
        return
    
    # Try signal performance analysis first
    print("1. Analyzing signal performance...")
    try:
        analyzer = SignalPerformanceAnalyzer(workspace)
        
        # Load signal events with prices
        signals_df = analyzer.load_signal_events()
        
        if signals_df.empty:
            print("   No signal events found in workspace")
        else:
            print(f"   Found {len(signals_df)} signal events")
            
            # Pair signals and calculate performance
            pairs = analyzer.pair_signals()
            print(f"   Created {len(pairs)} signal pairs")
            
            if pairs:
                metrics = analyzer.calculate_performance()
                
                # Print detailed report
                print("\n" + analyzer.get_summary_report())
                
                # Save analysis
                output_path = analyzer.save_analysis()
                print(f"\n   Analysis saved to: {output_path}")
    
    except Exception as e:
        print(f"   Signal analysis error: {e}")
    
    # Try analytics database if available
    analytics_db = workspace / 'analytics.duckdb'
    if analytics_db.exists():
        print("\n2. Analyzing using analytics database...")
        try:
            with AnalyticsWorkspace(workspace_path) as ws:
                # Get workspace summary
                summary = ws.summary()
                print(f"\n   Workspace Summary:")
                print(f"   - Run count: {summary['run_count']}")
                print(f"   - Total strategies: {summary['total_strategies']}")
                print(f"   - Strategy types: {summary['strategy_types']}")
                print(f"   - Total classifiers: {summary['total_classifiers']}")
                
                # Check for performance data
                print("\n   Querying strategy performance...")
                
                # Get top performing strategies
                top_strategies = ws.sql("""
                    SELECT 
                        strategy_id,
                        strategy_name,
                        strategy_type,
                        signal_file_path
                    FROM strategies
                    ORDER BY strategy_name
                    LIMIT 10
                """)
                
                if not top_strategies.empty:
                    print(f"\n   Found {len(top_strategies)} strategies in catalog")
                    print("\n   Sample strategies:")
                    for _, row in top_strategies.iterrows():
                        print(f"   - {row['strategy_name']} ({row['strategy_type']})")
                    
                    # Calculate performance for a sample strategy
                    sample_strategy = top_strategies.iloc[0]
                    print(f"\n   Calculating performance for: {sample_strategy['strategy_name']}")
                    
                    try:
                        # Load signals for the strategy
                        signals = ws.load_signals(sample_strategy['strategy_id'])
                        print(f"   Loaded {len(signals)} signals")
                        
                        # Get signal statistics
                        stats = ws.get_signal_statistics(sample_strategy['signal_file_path'])
                        print(f"\n   Signal Statistics:")
                        for key, value in stats.items():
                            print(f"   - {key}: {value}")
                    
                    except Exception as e:
                        print(f"   Error loading signals: {e}")
        
        except Exception as e:
            print(f"   Analytics database error: {e}")
    else:
        print("\n2. No analytics database found in workspace")
    
    # Check for raw parquet files
    print("\n3. Checking for raw data files...")
    
    # Look for event files
    event_files = list(workspace.glob("*/events.parquet"))
    if event_files:
        print(f"   Found {len(event_files)} event files")
        
        # Analyze first event file
        events_df = pd.read_parquet(event_files[0])
        print(f"\n   Sample event file: {event_files[0].parent.name}")
        print(f"   - Total events: {len(events_df)}")
        print(f"   - Event types: {events_df['event_type'].value_counts().to_dict()}")
    
    # Look for signal files
    signal_files = list(workspace.glob("*/signals.parquet"))
    if signal_files:
        print(f"\n   Found {len(signal_files)} signal files")
    
    # Look for metrics files
    metrics_files = list(workspace.glob("*/metrics.json"))
    if metrics_files:
        print(f"\n   Found {len(metrics_files)} metrics files")
        
        # Load and display first metrics file
        with open(metrics_files[0], 'r') as f:
            metrics = json.load(f)
        
        print(f"\n   Sample metrics from {metrics_files[0].parent.name}:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   - {key}: {value:.4f}" if isinstance(value, float) else f"   - {key}: {value}")
    
    # Look for traces directory
    traces_dir = workspace / 'traces'
    if traces_dir.exists():
        print(f"\n4. Found traces directory")
        symbols = list(traces_dir.iterdir())
        if symbols:
            print(f"   Symbols: {[s.name for s in symbols if s.is_dir()]}")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")


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
        analyze_workspace_signals(str(workspace_abs))
    else:
        print(f"Error: Could not find workspace at {workspace_path}")
        print(f"Looked in: {workspace_abs}")


if __name__ == "__main__":
    main()