#!/usr/bin/env python3
"""Check why true divergence strategy underperforms"""

import pandas as pd
from pathlib import Path
import numpy as np

# Load both strategy results
simple_workspace = "workspaces/signal_generation_3a27aae3"  # simple signals workspace
divergence_workspace = "workspaces/signal_generation_84f6c6a0"  # true divergence

def analyze_strategy(workspace_path, name):
    print(f"\n{'='*60}")
    print(f"Analyzing {name}")
    print("="*60)
    
    workspace = Path(workspace_path)
    signal_files = list(workspace.rglob("*.parquet"))
    
    if not signal_files:
        print(f"No signals found in {workspace_path}")
        return
    
    df = pd.read_parquet(signal_files[0])
    
    # Basic stats
    entries = df[df['val'] != 0]
    exits = df[df['val'] == 0]
    
    print(f"\nTotal signals: {len(df)}")
    print(f"Entries: {len(entries)}")
    print(f"Exits: {len(exits)}")
    
    # Analyze timing - are we late to the party?
    if 'idx' in df.columns:
        # Look at where signals occur in the dataset
        total_bars = 20448  # full test set
        entry_positions = entries['idx'].values / total_bars * 100
        
        print(f"\nEntry timing (% through dataset):")
        print(f"First entry: {entry_positions.min():.1f}%")
        print(f"Last entry: {entry_positions.max():.1f}%")
        print(f"Mean position: {entry_positions.mean():.1f}%")
        
        # Check signal spacing
        entry_indices = sorted(entries['idx'].values)
        if len(entry_indices) > 1:
            gaps = [entry_indices[i+1] - entry_indices[i] for i in range(len(entry_indices)-1)]
            print(f"\nBars between entries:")
            print(f"Mean: {np.mean(gaps):.1f}")
            print(f"Median: {np.median(gaps):.0f}")
            print(f"Min: {min(gaps)}")
            print(f"Max: {max(gaps)}")
    
    # Estimate holding periods
    if len(entries) > 0 and len(exits) > 0:
        # Simple estimation - match entries to next exit
        entry_indices = sorted(entries['idx'].values)
        exit_indices = sorted(exits['idx'].values)
        
        holding_periods = []
        for entry_idx in entry_indices:
            # Find next exit after this entry
            next_exits = [e for e in exit_indices if e > entry_idx]
            if next_exits:
                holding_periods.append(next_exits[0] - entry_idx)
        
        if holding_periods:
            print(f"\nEstimated holding periods (bars):")
            print(f"Mean: {np.mean(holding_periods):.1f}")
            print(f"Median: {np.median(holding_periods):.0f}")
            print(f"Min: {min(holding_periods)}")
            print(f"Max: {max(holding_periods)}")

# Analyze both strategies
if Path(divergence_workspace).exists():
    analyze_strategy(divergence_workspace, "True Divergence Strategy")
else:
    print(f"Divergence workspace not found: {divergence_workspace}")

# Note: The simple signals workspace was from an earlier run
print("\n" + "="*60)
print("Comparison Notes:")
print("="*60)
print("Simple Signals: 196 trades, 70.41% win rate, +10.7% return")
print("True Divergence: 71 trades, 53.5% win rate, -1.03% return")
print("\nKey differences:")
print("1. Simple signals trades 2.8x more frequently")
print("2. Simple signals reacts immediately to conditions")
print("3. True divergence waits for historical pattern confirmation")
print("4. True divergence may be entering after the initial move")