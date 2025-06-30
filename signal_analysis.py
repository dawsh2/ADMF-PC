#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

def main():
    # Read sweep signals (strategy 40 = period 15, std_dev 3.0)
    sweep_file = 'config/bollinger/results/20250623_062931/traces/bollinger_bands/SPY_5m_compiled_strategy_40.parquet'
    ensemble_file = 'config/ensemble/results/20250623_103142/traces/ensemble/SPY_5m_compiled_strategy_0.parquet'
    
    print("Reading signal files...")
    
    # Read sweep
    sweep_df = pd.read_parquet(sweep_file)
    print(f"\nParameter Sweep (period=15, std_dev=3.0):")
    print(f"  Total signal changes: {len(sweep_df)}")
    print(f"  Columns: {sweep_df.columns.tolist()}")
    
    # Read ensemble  
    ensemble_df = pd.read_parquet(ensemble_file)
    print(f"\nEnsemble (same params):")
    print(f"  Total signal changes: {len(ensemble_df)}")
    print(f"  Columns: {ensemble_df.columns.tolist()}")
    
    # Compare holding periods
    print("\n\nHOLDING PERIOD ANALYSIS:")
    print("-" * 50)
    
    # Sweep holding periods
    sweep_gaps = []
    for i in range(1, len(sweep_df)):
        gap = sweep_df.iloc[i]['idx'] - sweep_df.iloc[i-1]['idx']
        sweep_gaps.append(gap)
    
    # Ensemble holding periods
    ensemble_gaps = []
    for i in range(1, len(ensemble_df)):
        gap = ensemble_df.iloc[i]['idx'] - ensemble_df.iloc[i-1]['idx']
        ensemble_gaps.append(gap)
    
    if sweep_gaps:
        sweep_gaps = np.array(sweep_gaps)
        print(f"\nParameter Sweep:")
        print(f"  Average holding: {sweep_gaps.mean():.1f} bars")
        print(f"  Median holding: {np.median(sweep_gaps):.0f} bars")
        print(f"  Min holding: {sweep_gaps.min()} bars")
        print(f"  Max holding: {sweep_gaps.max()} bars")
        print(f"  1-bar exits: {(sweep_gaps == 1).sum()} ({(sweep_gaps == 1).mean()*100:.1f}%)")
    
    if ensemble_gaps:
        ensemble_gaps = np.array(ensemble_gaps)
        print(f"\nEnsemble:")
        print(f"  Average holding: {ensemble_gaps.mean():.1f} bars")
        print(f"  Median holding: {np.median(ensemble_gaps):.0f} bars")
        print(f"  Min holding: {ensemble_gaps.min()} bars")
        print(f"  Max holding: {ensemble_gaps.max()} bars")
        print(f"  1-bar exits: {(ensemble_gaps == 1).sum()} ({(ensemble_gaps == 1).mean()*100:.1f}%)")
    
    # Look at first few signals
    print("\n\nFIRST 10 SIGNAL CHANGES:")
    print("-" * 50)
    
    print("\nParameter Sweep:")
    for i in range(min(10, len(sweep_df))):
        row = sweep_df.iloc[i]
        print(f"  Bar {int(row['idx'])}: signal = {row['val']}")
        
    print("\nEnsemble:")
    for i in range(min(10, len(ensemble_df))):
        row = ensemble_df.iloc[i]
        print(f"  Bar {int(row['idx'])}: signal = {row['val']}")
    
    # Count signal types
    print("\n\nSIGNAL DISTRIBUTION:")
    print("-" * 50)
    
    sweep_vals = sweep_df['val'].value_counts()
    ensemble_vals = ensemble_df['val'].value_counts()
    
    print("\nParameter Sweep:")
    for val, count in sweep_vals.items():
        print(f"  Signal {val}: {count} occurrences")
        
    print("\nEnsemble:")
    for val, count in ensemble_vals.items():
        print(f"  Signal {val}: {count} occurrences")

if __name__ == "__main__":
    main()