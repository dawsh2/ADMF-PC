#!/usr/bin/env python3
"""
Quick check of grid search results
"""
import os
import pandas as pd
from pathlib import Path

def quick_check(workspace_path: str):
    """Quick check of what we have in the workspace."""
    
    workspace = Path(workspace_path)
    traces = workspace / "traces" / "SPY_1m"
    
    print(f"QUICK RESULTS CHECK: {workspace_path}")
    print("=" * 80)
    
    # Check signals
    signals_dir = traces / "signals"
    if signals_dir.exists():
        strategy_types = sorted([d.name for d in signals_dir.iterdir() if d.is_dir()])
        print(f"\nSTRATEGY TYPES ({len(strategy_types)}):")
        for i, st in enumerate(strategy_types, 1):
            files = list((signals_dir / st).glob("*.parquet"))
            print(f"  {i:2d}. {st:<30} → {len(files)} files")
        
        # Sample a few files to check structure
        print(f"\nSAMPLE SIGNAL FILES:")
        for strategy_type in strategy_types[:3]:
            strategy_dir = signals_dir / strategy_type
            files = list(strategy_dir.glob("*.parquet"))
            if files:
                try:
                    df = pd.read_parquet(files[0])
                    print(f"  {strategy_type}: {len(df)} rows, columns: {list(df.columns)}")
                    if len(df) > 0:
                        signal_counts = df['signal_value'].value_counts()
                        print(f"    Signal distribution: {dict(signal_counts)}")
                except Exception as e:
                    print(f"  {strategy_type}: ERROR - {e}")
    
    # Check classifiers  
    classifiers_dir = traces / "classifiers"
    if classifiers_dir.exists():
        classifier_types = sorted([d.name for d in classifiers_dir.iterdir() if d.is_dir()])
        print(f"\nCLASSIFIER TYPES ({len(classifier_types)}):")
        for i, ct in enumerate(classifier_types, 1):
            files = list((classifiers_dir / ct).glob("*.parquet"))
            print(f"  {i:2d}. {ct:<35} → {len(files)} files")
        
        # Sample classifier files
        print(f"\nSAMPLE CLASSIFIER FILES:")
        for classifier_type in classifier_types[:2]:
            classifier_dir = classifiers_dir / classifier_type
            files = list(classifier_dir.glob("*.parquet"))
            if files:
                try:
                    df = pd.read_parquet(files[0])
                    print(f"  {classifier_type}: {len(df)} rows, columns: {list(df.columns)}")
                    if len(df) > 0 and 'regime' in df.columns:
                        regime_counts = df['regime'].value_counts()
                        print(f"    Regime distribution: {dict(regime_counts)}")
                except Exception as e:
                    print(f"  {classifier_type}: ERROR - {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        workspace_path = sys.argv[1]
    else:
        workspace_path = "workspaces/expansive_grid_search_0397bd70"
    
    quick_check(workspace_path)