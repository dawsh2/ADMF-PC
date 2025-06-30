#!/usr/bin/env python3
"""
Verify if the filter change actually took effect.
"""

import pandas as pd
import json
from pathlib import Path

def verify_filter_change():
    print("=== VERIFYING FILTER CHANGE ===\n")
    
    # Check the last two runs
    results_dir = Path("/Users/daws/ADMF-PC/config/keltner/config_2826/results")
    
    # List all result directories
    result_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name != 'latest'])
    
    print("Recent runs:")
    for d in result_dirs[-3:]:
        metadata_path = d / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            signal_changes = metadata.get('stored_changes', 0)
            timestamp = d.name
            print(f"{timestamp}: {signal_changes} signals")
    
    # Check if config was actually used
    print("\nChecking config used in latest run...")
    
    # The fact that we got EXACTLY 726 signals both times suggests:
    print("\nPOSSIBLE ISSUES:")
    print("1. The config change didn't take effect")
    print("2. The data itself only has 726 Keltner band crossings")
    print("3. The filter is being applied elsewhere in the code")
    
    # Let's check without ANY filter
    print("\nCreating NO FILTER config to test...")
    
    no_filter_config = """# Test config with NO volatility filter
name: keltner_no_filter_test
data: SPY_5m  # Your test data

strategy:
  - keltner_bands:
      period: [30]
      multiplier: [1.0]
      # NO FILTER - should get maximum signals
"""
    
    config_path = results_dir.parent / "test_no_filter.yaml"
    with open(config_path, 'w') as f:
        f.write(no_filter_config)
    
    print(f"Created: {config_path}")
    print("\nIf this ALSO gives 726 signals, then:")
    print("- The test data simply has 726 Keltner crossings")
    print("- No amount of filter adjustment will help")
    print("- The strategy is fundamentally unsuitable for this test period")
    
    # Check signal patterns
    latest_signals = results_dir / "latest" / "traces" / "keltner_bands" / "SPY_5m_compiled_strategy_0.parquet"
    
    if latest_signals.exists():
        signals_df = pd.read_parquet(latest_signals)
        print(f"\nSignal analysis:")
        print(f"Total signal changes: {len(signals_df)}")
        print(f"Unique values: {signals_df['val'].unique()}")
        
        # Check signal distribution
        signal_counts = signals_df['val'].value_counts()
        print(f"\nSignal distribution:")
        for val, count in signal_counts.items():
            print(f"  Signal {val}: {count} occurrences")

if __name__ == "__main__":
    verify_filter_change()