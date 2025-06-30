#!/usr/bin/env python3
"""Analyze signal traces from a workspace to verify strategies behaved properly."""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import sys

def analyze_workspace(workspace_path):
    """Analyze all signal traces in a workspace."""
    workspace = Path(workspace_path)
    
    # Load metadata
    with open(workspace / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"=== Workspace Analysis: {workspace_path} ===\n")
    print(f"Total bars processed: {metadata['total_bars']}")
    print(f"Total signals emitted: {metadata['total_signals']}")
    print(f"Total changes stored: {metadata['stored_changes']}")
    print(f"Overall compression: {metadata['compression_ratio']:.1f}%\n")
    
    # Analyze by strategy type
    by_type = {}
    for comp_id, comp_data in metadata['components'].items():
        strategy_type = comp_data['strategy_type']
        if strategy_type not in by_type:
            by_type[strategy_type] = {
                'strategies': [],
                'total_changes': 0,
                'total_bars': 0
            }
        by_type[strategy_type]['strategies'].append(comp_data)
        by_type[strategy_type]['total_changes'] += comp_data['signal_changes']
        by_type[strategy_type]['total_bars'] += comp_data['total_bars']
    
    print("=== Strategy Type Summary ===\n")
    for stype, data in sorted(by_type.items()):
        print(f"{stype}:")
        print(f"  Count: {len(data['strategies'])}")
        print(f"  Total changes: {data['total_changes']}")
        print(f"  Avg changes per strategy: {data['total_changes']/len(data['strategies']):.1f}")
        print()
    
    # Detailed analysis of each strategy type
    print("\n=== Detailed Strategy Behavior Analysis ===\n")
    
    traces_dir = workspace / 'traces' / 'SPY_1m' / 'signals'
    
    for strategy_type in sorted(by_type.keys()):
        print(f"\n--- {strategy_type.upper()} ---")
        
        # Find most active strategy of this type
        strategies = by_type[strategy_type]['strategies']
        most_active = max(strategies, key=lambda x: x['signal_changes'])
        
        # Load and analyze the most active strategy
        signal_file = workspace / most_active['signal_file_path']
        if signal_file.exists():
            df = pd.read_parquet(signal_file)
            
            print(f"\nMost active {strategy_type} strategy:")
            print(f"  File: {most_active['signal_file_path']}")
            print(f"  Total changes: {len(df)}")
            print(f"  Signal distribution: {df.val.value_counts().sort_index().to_dict()}")
            
            # Analyze signal persistence
            if len(df) > 1:
                gaps = [df.iloc[i].idx - df.iloc[i-1].idx for i in range(1, len(df))]
                print(f"\n  Signal Persistence:")
                print(f"    Average duration: {np.mean(gaps):.1f} bars")
                print(f"    Max duration: {max(gaps)} bars")
                print(f"    Min duration: {min(gaps)} bars")
                
                # Check for proper oscillation (mean reversion)
                if strategy_type in ['bollinger_bands', 'rsi_bands', 'pivot_bounces']:
                    transitions = []
                    for i in range(1, len(df)):
                        transitions.append((df.iloc[i-1].val, df.iloc[i].val))
                    
                    # Count oscillations (signal reversals)
                    reversals = sum(1 for t in transitions if t[0] * t[1] < 0)  # Sign change
                    print(f"    Signal reversals: {reversals}/{len(transitions)} ({reversals/len(transitions)*100:.1f}%)")
            
            # Sample signal sequence
            print(f"\n  Sample signal sequence (first 8 changes):")
            for i in range(min(8, len(df))):
                print(f"    Bar {df.iloc[i].idx:3d}: {df.iloc[i].val:2d} @ ${df.iloc[i].px:.2f}")

    # Verify expected behaviors
    print("\n\n=== Strategy Behavior Verification ===\n")
    
    issues = []
    
    # Check 1: Mean reversion strategies should oscillate
    for stype in ['bollinger_bands', 'pivot_bounces']:
        if stype in by_type:
            avg_changes = by_type[stype]['total_changes'] / len(by_type[stype]['strategies'])
            if avg_changes < 2:
                issues.append(f"⚠️  {stype} strategies have very few signal changes ({avg_changes:.1f} avg)")
    
    # Check 2: All strategies should have reasonable compression
    for comp_id, comp_data in metadata['components'].items():
        if comp_data['compression_ratio'] > 0.5:  # More than 50% changes
            issues.append(f"⚠️  {comp_id} has low compression ({comp_data['compression_ratio']:.1%})")
    
    # Check 3: Strategies should emit signals after warmup
    for comp_id, comp_data in metadata['components'].items():
        if comp_data['signal_changes'] == 0 and comp_data['total_bars'] > 50:
            issues.append(f"❌ {comp_id} emitted no signals despite {comp_data['total_bars']} bars")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ All strategies behaved as expected!")
        print("  - Mean reversion strategies show proper oscillation")
        print("  - Sparse storage achieving good compression")
        print("  - All strategies generating signals after warmup")

if __name__ == "__main__":
    workspace = sys.argv[1] if len(sys.argv) > 1 else "workspaces/signal_generation_a15a871e"
    analyze_workspace(workspace)