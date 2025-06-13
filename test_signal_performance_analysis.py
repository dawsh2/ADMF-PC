#!/usr/bin/env python3
"""
Test signal performance analysis using stored signals.

This script demonstrates how to analyze signal performance from
the hierarchical storage without requiring trade execution.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from src.analytics.signal_performance_analyzer import SignalPerformanceAnalyzer, analyze_signal_performance


def main():
    """Run signal performance analysis on stored data."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Find most recent workspace or use flat structure
    workspaces_dir = Path('workspaces')
    if not workspaces_dir.exists():
        print("No workspaces directory found. Run signal generation first.")
        return
    
    # For now, use the workspaces directory itself as the workspace
    # since containers are being created at the top level
    workspace_path = workspaces_dir
    
    print(f"\nAnalyzing signals from workspace: {workspace_path}")
    print("=" * 60)
    
    # List all container directories
    containers = [d for d in workspace_path.iterdir() if d.is_dir()]
    print(f"\nFound {len(containers)} containers")
    
    # Show recent containers
    recent_containers = sorted(containers, key=lambda x: x.stat().st_mtime, reverse=True)[:10]
    print("\nRecent containers:")
    for container in recent_containers:
        print(f"  - {container.name}")
    
    # Create analyzer
    analyzer = SignalPerformanceAnalyzer(workspace_path)
    
    # Load signal events
    signals_df = analyzer.load_signal_events()
    
    if signals_df.empty:
        print("No signals found in workspace")
        return
    
    print(f"\nLoaded {len(signals_df)} signals")
    
    # Show signal distribution
    if 'strategy_name' in signals_df.columns:
        print("\nSignals by strategy:")
        print(signals_df['strategy_name'].value_counts())
    
    if 'symbol' in signals_df.columns:
        print("\nSignals by symbol:")
        print(signals_df['symbol'].value_counts())
    
    if 'direction' in signals_df.columns:
        print("\nSignals by direction:")
        print(signals_df['direction'].value_counts())
    
    # Pair signals
    print("\n" + "=" * 60)
    print("Pairing entry/exit signals...")
    signal_pairs = analyzer.pair_signals()
    
    print(f"Created {len(signal_pairs)} signal pairs")
    
    if signal_pairs:
        # Show some example pairs
        print("\nExample signal pairs (first 3):")
        for i, pair in enumerate(signal_pairs[:3]):
            print(f"\nPair {i+1}:")
            print(f"  Strategy: {pair['strategy']}")
            print(f"  Symbol: {pair['symbol']}")
            print(f"  Direction: {pair['direction']}")
            print(f"  Entry: {pair['entry_price']:.2f} at {pair['entry_time']}")
            print(f"  Exit: {pair['exit_price']:.2f} at {pair['exit_time']}")
            print(f"  P&L: {pair['pnl']:.2f} ({pair['pnl_pct']:.2%})")
            print(f"  Holding period: {pair['holding_period']/3600:.1f} hours")
    
    # Calculate performance
    print("\n" + "=" * 60)
    print("Calculating performance metrics...")
    metrics = analyzer.calculate_performance()
    
    # Print summary report
    print(analyzer.get_summary_report())
    
    # Save analysis
    output_path = analyzer.save_analysis()
    print(f"\nAnalysis saved to: {output_path}")
    
    # Also save to a more accessible location
    summary_path = Path('results/signal_performance_summary.json')
    summary_path.parent.mkdir(exist_ok=True)
    
    import json
    with open(summary_path, 'w') as f:
        json.dump({
            'workspace': str(workspace_path),
            'metrics': metrics,
            'signal_pairs_count': len(signal_pairs),
            'report': analyzer.get_summary_report()
        }, f, indent=2, default=str)
    
    print(f"Summary also saved to: {summary_path}")


if __name__ == "__main__":
    main()