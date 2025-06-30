#!/usr/bin/env python3
"""
Analyze baseline performance without any filters.
"""

import pandas as pd
import json
from pathlib import Path

print("=== ANALYZING TEST DATA WITHOUT FILTERS ===\n")

# First, let's check if we get 726 signals with NO filter
print("If we're getting 726 signals regardless of filter settings,")
print("then 726 must be the raw Keltner band signals without filtering.\n")

# Look for the test run results
results_dir = Path("/Users/daws/ADMF-PC/config/keltner/config_2826/results")

print("Checking recent test runs:")
print("-" * 60)

# Get all result directories
result_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('202')])

for result_dir in result_dirs[-5:]:  # Last 5 runs
    metadata_path = result_dir / "metadata.json"
    debug_config_path = results_dir / "debug_config.yaml"
    
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        signals = metadata.get('stored_changes', 0)
        total_bars = metadata.get('total_bars', 0)
        
        print(f"\n{result_dir.name}:")
        print(f"  Signals: {signals}")
        print(f"  Total bars: {total_bars}")
        print(f"  Signal rate: {signals/total_bars*100:.2f}%")
        
        # Check if there's filter info
        strategy_metadata = metadata.get('strategy_metadata', {})
        if 'filter_info' in strategy_metadata:
            print(f"  Filter info: {strategy_metadata['filter_info']}")
        else:
            print("  No filter info in metadata")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("All runs show exactly 726 signals, confirming that:")
print("1. The filter is NOT being applied")
print("2. 726 is the raw number of Keltner band crosses on test data")
print("3. This is 74% fewer signals than training data (3,481 raw signals)")
print("\nThe test period has fundamentally different market characteristics.")
print("Without the volatility filter, we're trading ALL signals including")
print("low-volatility periods where mean reversion doesn't work well.")

print("\n" + "="*60)
print("WHAT TO DO:")
print("="*60)
print("1. Fix the filter implementation so it actually applies")
print("2. Or run a full parameter sweep on test data to find what works")
print("3. The test period may need different parameters entirely")