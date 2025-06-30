#!/usr/bin/env python3
"""
Quick analysis script for command-line usage.

Examples:
    # Interactive shell with pre-loaded data
    python -m src.analytics.analyze results/run_20250623_143030
    
    # Quick summary
    python -m src.analytics.analyze results/latest --summary
    
    # Find ensemble
    python -m src.analytics.analyze results/latest --ensemble
    
    # Export top strategies
    python -m src.analytics.analyze results/latest --export top_strategies.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.analytics.interactive import AnalysisWorkspace, QueryLibrary


def main():
    parser = argparse.ArgumentParser(description="Interactive analysis for ADMF-PC")
    parser.add_argument("run_path", help="Path to backtest run directory")
    parser.add_argument("--summary", action="store_true", help="Show run summary and exit")
    parser.add_argument("--ensemble", action="store_true", help="Find optimal ensemble")
    parser.add_argument("--export", help="Export top strategies to CSV")
    parser.add_argument("--query", help="Execute custom DuckDB query")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive shell")
    
    args = parser.parse_args()
    
    # Create workspace and load run
    workspace = AnalysisWorkspace()
    try:
        run = workspace.load_run(args.run_path)
    except Exception as e:
        print(f"Error loading run: {e}")
        sys.exit(1)
    
    # Handle different modes
    if args.summary:
        print_summary(run)
    
    elif args.ensemble:
        find_ensemble(workspace, run)
    
    elif args.export:
        export_strategies(workspace, run, args.export)
    
    elif args.query:
        execute_query(run, args.query)
    
    elif args.interactive:
        start_interactive(workspace, run)
    
    else:
        # Default: show summary and top strategies
        print_summary(run)
        print("\nTop 10 Strategies:")
        print_top_strategies(workspace, run, 10)


def print_summary(run):
    """Print run summary."""
    print("\nRun Summary:")
    print("-" * 50)
    for key, value in run.summary.items():
        print(f"  {key}: {value}")


def print_top_strategies(workspace, run, n=10):
    """Print top N strategies."""
    top = workspace.top_strategies(run, n=n)
    print(top[['strategy_type', 'strategy_hash', 'sharpe_ratio', 'total_return']].to_string())


def find_ensemble(workspace, run):
    """Find and display optimal ensemble."""
    print("\nFinding optimal ensemble...")
    ensemble = workspace.find_ensemble(run, size=5)
    
    print(f"\nOptimal Ensemble:")
    print(f"  Average Sharpe: {ensemble['avg_sharpe']:.2f}")
    print(f"  Max Correlation: {ensemble['max_correlation']:.2f}")
    print("\nEnsemble Strategies:")
    print(ensemble['strategies'][['strategy_type', 'sharpe_ratio', 'total_return']].to_string())


def export_strategies(workspace, run, output_path):
    """Export top strategies to CSV."""
    top = workspace.top_strategies(run, n=50)
    top.to_csv(output_path, index=False)
    print(f"Exported {len(top)} strategies to {output_path}")


def execute_query(run, query):
    """Execute custom query and display results."""
    try:
        result = run.query(query)
        print(result.to_string())
    except Exception as e:
        print(f"Query error: {e}")


def start_interactive(workspace, run):
    """Start interactive Python shell with pre-loaded data."""
    import IPython
    
    # Pre-load useful variables
    namespace = {
        'workspace': workspace,
        'run': run,
        'strategies': run.strategies,
        'QueryLibrary': QueryLibrary,
        'top': workspace.top_strategies(run, n=20),
        'query': run.query,  # Shortcut for queries
    }
    
    # Print welcome message
    print("\nInteractive Analysis Session")
    print("-" * 50)
    print("Available variables:")
    print("  workspace  - AnalysisWorkspace instance")
    print("  run        - Current BacktestRun")
    print("  strategies - DataFrame of all strategies")
    print("  top        - Top 20 strategies")
    print("  query()    - Execute DuckDB queries")
    print("\nExample commands:")
    print("  ensemble = workspace.find_ensemble(run)")
    print("  high_sharpe = query('SELECT * FROM strategies WHERE sharpe_ratio > 2')")
    print("  signal_freq = QueryLibrary.signal_frequency(run)")
    print("-" * 50)
    
    # Start IPython shell
    IPython.start_ipython(argv=[], user_ns=namespace)


if __name__ == "__main__":
    main()