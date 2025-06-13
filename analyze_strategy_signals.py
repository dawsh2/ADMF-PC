#!/usr/bin/env python3
"""
Analyze strategy performance from sparse signal files.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime

def load_sparse_signals(filepath: str) -> Dict:
    """Load sparse signal data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def reconstruct_signal_series(sparse_data: Dict) -> pd.DataFrame:
    """Reconstruct full signal series from sparse representation."""
    changes = sparse_data['changes']
    total_bars = sparse_data['metadata']['total_bars']
    
    if not changes:
        return pd.DataFrame()
    
    # Create DataFrame from changes
    df_changes = pd.DataFrame(changes)
    df_changes['timestamp'] = pd.to_datetime(df_changes['ts'])
    df_changes = df_changes.sort_values('idx')
    
    # Reconstruct full signal series
    signals = []
    current_signal = 0
    change_idx = 0
    
    for bar_idx in range(total_bars):
        # Check if we have a signal change at this index
        if change_idx < len(changes) and changes[change_idx]['idx'] == bar_idx:
            current_signal = changes[change_idx]['val']
            signals.append({
                'bar_idx': bar_idx,
                'signal': current_signal,
                'timestamp': changes[change_idx]['ts'],
                'strategy': changes[change_idx]['strat'],
                'symbol': changes[change_idx]['sym']
            })
            change_idx += 1
        else:
            # Carry forward the previous signal
            if signals:
                signals.append({
                    'bar_idx': bar_idx,
                    'signal': current_signal,
                    'timestamp': signals[-1]['timestamp'],  # Use last known timestamp
                    'strategy': signals[-1]['strategy'],
                    'symbol': signals[-1]['symbol']
                })
    
    return pd.DataFrame(signals)

def calculate_strategy_metrics(signal_df: pd.DataFrame) -> Dict:
    """Calculate performance metrics for a strategy."""
    if signal_df.empty:
        return {}
    
    # Calculate position changes
    signal_df['position_change'] = signal_df['signal'].diff() != 0
    position_changes = signal_df[signal_df['position_change']]
    
    # Calculate trade statistics
    num_trades = len(position_changes) - 1  # Exclude first position
    
    # Calculate position durations
    position_durations = []
    if len(position_changes) > 1:
        for i in range(len(position_changes) - 1):
            duration = position_changes.iloc[i+1]['bar_idx'] - position_changes.iloc[i]['bar_idx']
            position_durations.append(duration)
    
    # Count position types
    long_positions = (signal_df['signal'] == 1).sum()
    short_positions = (signal_df['signal'] == -1).sum()
    flat_positions = (signal_df['signal'] == 0).sum()
    
    # Calculate signal statistics
    total_bars = len(signal_df)
    signal_frequency = num_trades / total_bars if total_bars > 0 else 0
    
    metrics = {
        'total_bars': total_bars,
        'num_trades': num_trades,
        'long_bars': long_positions,
        'short_bars': short_positions,
        'flat_bars': flat_positions,
        'long_percentage': (long_positions / total_bars * 100) if total_bars > 0 else 0,
        'short_percentage': (short_positions / total_bars * 100) if total_bars > 0 else 0,
        'flat_percentage': (flat_positions / total_bars * 100) if total_bars > 0 else 0,
        'signal_frequency': signal_frequency,
        'avg_position_duration': sum(position_durations) / len(position_durations) if position_durations else 0,
        'min_position_duration': min(position_durations) if position_durations else 0,
        'max_position_duration': max(position_durations) if position_durations else 0,
    }
    
    return metrics

def analyze_signal_files(workspace_dir: str) -> None:
    """Analyze all signal files in a workspace directory."""
    workspace_path = Path(workspace_dir)
    signal_files = list(workspace_path.glob("signals_strategy_*.json"))
    
    if not signal_files:
        print(f"No signal files found in {workspace_dir}")
        return
    
    print(f"\nAnalyzing {len(signal_files)} signal files in {workspace_dir}\n")
    print("=" * 80)
    
    all_metrics = []
    
    for signal_file in sorted(signal_files):
        print(f"\nAnalyzing: {signal_file.name}")
        print("-" * 40)
        
        # Load sparse signal data
        sparse_data = load_sparse_signals(signal_file)
        metadata = sparse_data['metadata']
        
        # Extract strategy info
        strategies = metadata.get('strategies', {})
        for strategy_id, strategy_info in strategies.items():
            print(f"Strategy ID: {strategy_id}")
            
            # Get pre-calculated statistics from metadata
            signal_stats = metadata.get('signal_statistics', {})
            by_strategy_stats = signal_stats.get('by_strategy', {}).get(strategy_id, {})
            
            # Reconstruct full signal series for detailed analysis
            signal_df = reconstruct_signal_series(sparse_data)
            
            # Calculate additional metrics
            metrics = calculate_strategy_metrics(signal_df)
            
            # Display results
            print(f"  Total bars processed: {metrics.get('total_bars', 'N/A')}")
            print(f"  Number of trades: {metrics.get('num_trades', 'N/A')}")
            print(f"  Signal frequency: {metrics.get('signal_frequency', 0):.2%}")
            print(f"  Compression ratio: {metadata.get('compression_ratio', 0):.2%}")
            print()
            print(f"  Position breakdown:")
            print(f"    Long:  {metrics.get('long_bars', 0):4d} bars ({metrics.get('long_percentage', 0):5.1f}%)")
            print(f"    Short: {metrics.get('short_bars', 0):4d} bars ({metrics.get('short_percentage', 0):5.1f}%)")
            print(f"    Flat:  {metrics.get('flat_bars', 0):4d} bars ({metrics.get('flat_percentage', 0):5.1f}%)")
            print()
            print(f"  Position durations:")
            print(f"    Average: {metrics.get('avg_position_duration', 0):.1f} bars")
            print(f"    Min:     {metrics.get('min_position_duration', 0)} bars")
            print(f"    Max:     {metrics.get('max_position_duration', 0)} bars")
            
            # Store for comparison
            all_metrics.append({
                'strategy_id': strategy_id,
                'file': signal_file.name,
                **metrics
            })
    
    # Summary comparison
    if len(all_metrics) > 1:
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON")
        print("=" * 80)
        
        df_metrics = pd.DataFrame(all_metrics)
        
        # Create comparison table
        comparison_cols = ['strategy_id', 'num_trades', 'signal_frequency', 
                          'long_percentage', 'short_percentage', 'avg_position_duration']
        
        print("\nKey Metrics Comparison:")
        print(df_metrics[comparison_cols].to_string(index=False, float_format='%.2f'))
        
        # Find best/worst performers
        print("\n" + "-" * 40)
        print("Performance Highlights:")
        print(f"  Most active strategy: {df_metrics.loc[df_metrics['num_trades'].idxmax(), 'strategy_id']} "
              f"({df_metrics['num_trades'].max()} trades)")
        print(f"  Most patient strategy: {df_metrics.loc[df_metrics['avg_position_duration'].idxmax(), 'strategy_id']} "
              f"({df_metrics['avg_position_duration'].max():.1f} bars avg)")
        print(f"  Most directional: {df_metrics.loc[(df_metrics['long_percentage'] - df_metrics['short_percentage']).abs().idxmax(), 'strategy_id']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        workspace_dir = sys.argv[1]
    else:
        # Default to most recent workspace
        workspace_root = Path("workspaces/tmp")
        if workspace_root.exists():
            workspaces = sorted([d for d in workspace_root.iterdir() if d.is_dir()])
            if workspaces:
                workspace_dir = str(workspaces[-1])
            else:
                print("No workspaces found in workspaces/tmp/")
                sys.exit(1)
        else:
            print("Workspace directory not found")
            sys.exit(1)
    
    analyze_signal_files(workspace_dir)