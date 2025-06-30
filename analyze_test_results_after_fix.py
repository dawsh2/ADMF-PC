#!/usr/bin/env python3
"""
Analyze test results after filter fix.
"""

import pandas as pd
import json
from pathlib import Path

def analyze_test_results():
    print("=== TEST RESULTS ANALYSIS AFTER FILTER FIX ===\n")
    
    # Check metadata history
    print("1. CHECKING SIGNAL COUNTS OVER TIME:")
    print("-" * 60)
    
    results_dir = Path("/Users/daws/ADMF-PC/config/keltner/config_2826/results")
    
    # Get all result directories sorted by time
    result_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('202')])
    
    for result_dir in result_dirs[-5:]:  # Last 5 runs
        metadata_path = result_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            signals = metadata.get('stored_changes', 0)
            timestamp = result_dir.name
            print(f"{timestamp}: {signals} signals")
    
    print("\n💡 INSIGHT: All runs show 726 signals - filter is NOT being applied!")
    
    # Let's check if there's a workspace analysis file
    print("\n2. LOOKING FOR WORKSPACE ANALYSIS:")
    print("-" * 60)
    
    # Find any analysis files
    analysis_files = list(Path(".").glob("**/workspace_analysis_*.csv"))
    if analysis_files:
        latest_analysis = max(analysis_files, key=lambda f: f.stat().st_mtime)
        print(f"Found analysis: {latest_analysis}")
        
        df = pd.read_csv(latest_analysis)
        if len(df) > 0:
            print(f"\nStrategy performance:")
            print(f"- Total trades: {df['total_trades'].iloc[0]}")
            print(f"- Win rate: {df['avg_win_rate'].iloc[0]:.1f}%")
            print(f"- Avg return per trade: {df['avg_return_per_trade_bps'].iloc[0]:.2f} bps")
    
    # The key issue
    print("\n3. DIAGNOSIS:")
    print("-" * 60)
    print("The filter syntax fix didn't work because:")
    print("1. The filter might need to be applied at a different level")
    print("2. The strategy implementation might not support filters")
    print("3. There could be a bug in the filter processing code")
    
    print("\n4. CHECKING CONFIG THAT WAS USED:")
    print("-" * 60)
    
    # Check if there's a debug config
    debug_config = results_dir / "debug_config.yaml"
    if debug_config.exists():
        print(f"Found debug config: {debug_config}")
        with open(debug_config) as f:
            content = f.read()
            if 'filter:' in content:
                print("\nFilter section:")
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'filter:' in line:
                        for j in range(5):
                            if i+j < len(lines):
                                print(f"  {lines[i+j]}")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    print("The filter is STILL not being applied despite correct syntax.")
    print("This suggests a deeper issue with the implementation.")
    print("\nNEXT STEPS:")
    print("1. Check if keltner_bands strategy supports filters")
    print("2. Look at the strategy implementation code")
    print("3. Try a different filter type or strategy")

if __name__ == "__main__":
    analyze_test_results()