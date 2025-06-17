#!/usr/bin/env python3
"""
Investigate why all ensemble strategies are producing identical signals.
This suggests there might be an issue with the ensemble implementation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def investigate_identical_signals():
    workspace_path = Path("workspaces/two_layer_regime_ensemble_v1_4f71d9e1")
    traces_path = workspace_path / "traces" / "SPY_1m"
    
    print("=== INVESTIGATING IDENTICAL SIGNALS ===\n")
    
    # Load all strategy files
    signal_dirs = [
        traces_path / "signals" / "ma_crossover",
        traces_path / "signals" / "regime"
    ]
    
    all_data = {}
    
    for signal_dir in signal_dirs:
        if signal_dir.exists():
            for file_path in signal_dir.glob("*.parquet"):
                strategy_name = file_path.stem.replace("SPY_", "")
                data = pd.read_parquet(file_path)
                
                # Standardize columns
                data['timestamp'] = pd.to_datetime(data['ts'])
                data['signal'] = data['val']
                data['price'] = data['px']
                
                all_data[strategy_name] = data
                
    print(f"Loaded {len(all_data)} strategies\n")
    
    # Check if all strategies have identical timestamps
    print("=== TIMESTAMP ANALYSIS ===")
    timestamps_by_strategy = {}
    for name, data in all_data.items():
        timestamps_by_strategy[name] = set(data['timestamp'])
        
    strategy_names = list(timestamps_by_strategy.keys())
    print(f"Strategy timestamp counts:")
    for name in strategy_names:
        print(f"  - {name}: {len(timestamps_by_strategy[name])} timestamps")
        
    # Check intersection
    common_timestamps = timestamps_by_strategy[strategy_names[0]]
    for name in strategy_names[1:]:
        common_timestamps = common_timestamps.intersection(timestamps_by_strategy[name])
        
    print(f"\nCommon timestamps: {len(common_timestamps)}")
    
    # Check if signals are identical at each timestamp
    print("\n=== SIGNAL COMPARISON ===")
    
    # Create aligned dataframes
    aligned_data = {}
    for name, data in all_data.items():
        # Sort by timestamp and create series
        sorted_data = data.sort_values('timestamp')
        aligned_data[name] = {
            'timestamp': sorted_data['timestamp'].values,
            'signal': sorted_data['signal'].values,
            'price': sorted_data['price'].values
        }
        
    # Compare first few signals
    print("First 10 signal values for each strategy:")
    for name, data in aligned_data.items():
        signals_str = ', '.join([str(s) for s in data['signal'][:10]])
        print(f"  {name}: [{signals_str}]")
        
    print("\nLast 10 signal values for each strategy:")
    for name, data in aligned_data.items():
        signals_str = ', '.join([str(s) for s in data['signal'][-10:]])
        print(f"  {name}: [{signals_str}]")
        
    # Check if all signals are identical
    print("\n=== IDENTITY CHECK ===")
    base_strategy = strategy_names[0]
    base_signals = aligned_data[base_strategy]['signal']
    
    all_identical = True
    for name in strategy_names[1:]:
        current_signals = aligned_data[name]['signal']
        
        # Check if arrays are identical
        if len(base_signals) == len(current_signals):
            identical = np.array_equal(base_signals, current_signals)
            print(f"{base_strategy} vs {name}: {'IDENTICAL' if identical else 'DIFFERENT'}")
            
            if not identical:
                all_identical = False
                # Show some differences
                differences = np.where(base_signals != current_signals)[0]
                print(f"  Differences at indices: {differences[:10]}...")
        else:
            print(f"{base_strategy} vs {name}: DIFFERENT LENGTHS ({len(base_signals)} vs {len(current_signals)})")
            all_identical = False
            
    print(f"\nAll strategies identical: {all_identical}")
    
    # Check if this is due to sparse storage (only signal changes stored)
    print("\n=== SPARSE STORAGE ANALYSIS ===")
    
    for name, data in all_data.items():
        sorted_data = data.sort_values('timestamp')
        
        # Check if signals change
        signal_changes = (sorted_data['signal'] != sorted_data['signal'].shift(1)).sum()
        total_signals = len(sorted_data)
        
        print(f"{name}:")
        print(f"  Total signals: {total_signals}")
        print(f"  Signal changes: {signal_changes}")
        print(f"  Change rate: {signal_changes/total_signals*100:.2f}%")
        
        # Show signal transitions
        transitions = []
        for i in range(min(20, len(sorted_data)-1)):
            current = sorted_data['signal'].iloc[i]
            next_signal = sorted_data['signal'].iloc[i+1]
            transitions.append(f"{current}→{next_signal}")
            
        print(f"  First 20 transitions: {', '.join(transitions)}")
        print()
        
    # Analyze regime data for comparison
    print("=== REGIME DATA ANALYSIS ===")
    regime_file = traces_path / "classifiers" / "regime" / "SPY_market_regime_detector.parquet"
    if regime_file.exists():
        regime_data = pd.read_parquet(regime_file)
        regime_data['timestamp'] = pd.to_datetime(regime_data['ts'])
        regime_data = regime_data.sort_values('timestamp')
        
        print(f"Regime data:")
        print(f"  Total regime points: {len(regime_data)}")
        
        regime_changes = (regime_data['val'] != regime_data['val'].shift(1)).sum()
        print(f"  Regime changes: {regime_changes}")
        print(f"  Change rate: {regime_changes/len(regime_data)*100:.2f}%")
        
        # Show first few regime values
        first_regimes = regime_data['val'].head(10).tolist()
        print(f"  First 10 regimes: {first_regimes}")
        
        # Check if regime changes correspond to strategy signal changes
        print(f"\nRegime vs Strategy timing:")
        print(f"  Regime timespan: {regime_data['timestamp'].min()} to {regime_data['timestamp'].max()}")
        
        for name, data in all_data.items():
            strategy_data = data.sort_values('timestamp')
            print(f"  {name}: {strategy_data['timestamp'].min()} to {strategy_data['timestamp'].max()}")
            break  # They're all the same, so just show one
            
    # Check metadata for ensemble configuration
    print("\n=== ENSEMBLE CONFIGURATION CHECK ===")
    metadata_file = workspace_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        if 'config' in metadata:
            config = metadata['config']
            
            print("Ensemble configuration:")
            for key, value in config.items():
                if isinstance(value, dict) and len(str(value)) > 100:
                    print(f"  {key}: {type(value).__name__} with {len(value)} items")
                else:
                    print(f"  {key}: {value}")
                    
        # Check for any indication of identical strategies
        print(f"\nChecking for configuration issues...")
        
        # Look for baseline strategies
        if 'baseline_strategies' in config:
            baseline = config['baseline_strategies']
            print(f"Baseline strategy config: {baseline}")
            
        # Look for regime boosters
        if 'regime_boosters' in config:
            boosters = config['regime_boosters']
            print(f"Regime boosters:")
            for regime, strategies in boosters.items():
                print(f"  {regime}: {len(strategies)} strategies")


if __name__ == "__main__":
    investigate_identical_signals()