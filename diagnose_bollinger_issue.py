#!/usr/bin/env python3
"""
Diagnose the discrepancy between parameter sweep and ensemble results.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_signals(signals_path, label):
    """Analyze signal patterns to understand the issue."""
    signals = pd.read_parquet(signals_path)
    
    print(f"\n{label}:")
    print(f"  Total signal changes: {len(signals)}")
    
    # Look at signal patterns
    signal_values = signals['val'].value_counts().sort_index()
    print(f"  Signal distribution: {dict(signal_values)}")
    
    # Analyze holding periods
    holding_periods = []
    last_idx = 0
    last_val = 0
    
    for _, row in signals.iterrows():
        if row['val'] != last_val and last_val != 0:
            holding_period = row['idx'] - last_idx
            holding_periods.append(holding_period)
        if row['val'] != 0:
            last_idx = row['idx']
            last_val = row['val']
    
    if holding_periods:
        print(f"  Average holding period: {np.mean(holding_periods):.1f} bars")
        print(f"  Median holding period: {np.median(holding_periods):.0f} bars")
        print(f"  Min/Max holding: {min(holding_periods)}/{max(holding_periods)} bars")
    
    # Look at first 10 signals
    print(f"\n  First 10 signal changes:")
    for i, row in signals.head(10).iterrows():
        print(f"    Bar {int(row['idx'])}: {row['val']}")
    
    return signals

def main():
    print("Diagnosing Bollinger Bands Performance Discrepancy")
    print("=" * 60)
    
    # Check ensemble config for exit_threshold
    print("\n1. Checking ensemble configuration:")
    import yaml
    with open('config/ensemble/config.yaml', 'r') as f:
        ensemble_config = yaml.safe_load(f)
    
    strategy_config = ensemble_config['strategy'][0]['bollinger_bands']
    print(f"   Period: {strategy_config.get('period', 'default')}")
    print(f"   Std dev: {strategy_config.get('std_dev', 'default')}")
    print(f"   Exit threshold: {strategy_config.get('exit_threshold', 'NOT SET - using default 0.001')}")
    
    # Check bollinger sweep config
    print("\n2. Checking bollinger sweep configuration:")
    bollinger_config_path = Path('config/bollinger/config.yaml')
    if bollinger_config_path.exists():
        with open(bollinger_config_path, 'r') as f:
            bollinger_config = yaml.safe_load(f)
        
        bb_strategy = bollinger_config['strategy'][0]['bollinger_bands']
        print(f"   Exit threshold: {bb_strategy.get('exit_threshold', 'NOT SET - using default 0.001')}")
    
    # Compare signals
    print("\n3. Comparing signal patterns:")
    
    # Find the parameter sweep signal file for period=15, std_dev=3.0
    # Strategy number = (period-10) * 7 + std_dev_index
    # period=15 -> 5, std_dev=3.0 -> index 5
    strategy_num = (15-10) * 7 + 5  # = 5*7 + 5 = 40
    
    sweep_path = Path(f"config/bollinger/results/latest/traces/bollinger_bands/SPY_5m_compiled_strategy_{strategy_num}.parquet")
    ensemble_path = Path("config/ensemble/results/latest/traces/ensemble/SPY_5m_compiled_strategy_0.parquet")
    
    if sweep_path.exists():
        sweep_signals = analyze_signals(sweep_path, f"Parameter sweep (strategy {strategy_num})")
    else:
        print(f"\n❌ Sweep signal file not found: {sweep_path}")
        # Try to find it
        traces_dir = Path("config/bollinger/results/latest/traces")
        if traces_dir.exists():
            print("\nAvailable trace directories:")
            for d in traces_dir.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
                    # Count files
                    files = list(d.glob("*.parquet"))
                    print(f"    Files: {len(files)}")
    
    if ensemble_path.exists():
        ensemble_signals = analyze_signals(ensemble_path, "Ensemble run")
    
    # The issue is likely the exit threshold
    print("\n4. Diagnosis:")
    print("-" * 50)
    print("The 1.4% win rate indicates trades are exiting immediately at tiny losses.")
    print("This happens when exit_threshold is too tight (0.001 = 0.1%).")
    print("\nPossible causes:")
    print("1. Parameter sweep might have used a different exit_threshold")
    print("2. The exit logic might be implemented differently")
    print("3. Data alignment issues causing incorrect price comparisons")
    
    print("\n5. Solution:")
    print("Add exit_threshold to your ensemble config:")
    print("  {bollinger_bands: {period: 15, std_dev: 3.0, exit_threshold: 0.003}}")

if __name__ == "__main__":
    main()