#!/usr/bin/env python3
"""
Analyze why the 2826 strategy produced 0 signals on test data.
"""

import pandas as pd
import json
from pathlib import Path
import glob

def analyze_test_results():
    print("=== ANALYZING TEST DATA RESULTS ===\n")
    
    # Check the latest test run
    test_result_path = "/Users/daws/ADMF-PC/config/keltner/config_2826/results/latest"
    metadata_path = Path(test_result_path) / "metadata.json"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("Test Run Summary:")
    print("-" * 50)
    print(f"Total bars: {metadata['total_bars']}")
    print(f"Total signals: {metadata['total_signals']}")
    print(f"Stored changes: {metadata['stored_changes']}")
    print(f"Components: {len(metadata['components'])}")
    
    if metadata['total_signals'] == 0:
        print("\n⚠️  WARNING: No signals generated!")
        print("\nPossible reasons:")
        print("1. Test data period has different characteristics")
        print("2. Volatility filter is too restrictive for test period")
        print("3. Data format/loading issue")
        print("4. Strategy compilation issue")
    
    # Check if signal files exist
    traces_dir = Path(test_result_path) / "traces" / "keltner_bands"
    if traces_dir.exists():
        signal_files = list(traces_dir.glob("*.parquet"))
        print(f"\nSignal files found: {len(signal_files)}")
        
        if signal_files:
            # Try to load one
            try:
                test_signals = pd.read_parquet(signal_files[0])
                print(f"Signal file contains {len(test_signals)} records")
            except Exception as e:
                print(f"Error loading signal file: {e}")
    else:
        print("\nNo traces directory found")
    
    # Compare training vs test data characteristics
    print("\n" + "="*60)
    print("COMPARING DATA CHARACTERISTICS")
    print("="*60)
    
    # Training data info
    print("\nTraining data (SPY_5m):")
    print("  Period: ~1 year")
    print("  Bars: 16,612")
    print("  Signals generated: 2,826")
    print("  Signal rate: 17.0%")
    
    # Test data info
    print("\nTest data:")
    print(f"  Bars: {metadata['total_bars']}")
    print("  Signals generated: 0")
    print("  Signal rate: 0%")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. Check volatility levels in test period:")
    print("   - The strategy requires ATR > 1.1x average")
    print("   - Test period might be less volatile")
    
    print("\n2. Verify data loading:")
    print("   - Ensure test data has same format as training")
    print("   - Check for missing OHLCV columns")
    
    print("\n3. Try relaxing the filter:")
    print("   - Lower volatility threshold to 1.0 or 0.9")
    print("   - Or remove filter temporarily to verify strategy works")
    
    print("\n4. Debug the strategy execution:")
    print("   - Add logging to see why signals aren't generated")
    print("   - Check if Keltner bands are being calculated correctly")
    
    # Create a debug config
    print("\n" + "="*60)
    print("CREATING DEBUG CONFIG")
    print("="*60)
    
    debug_config = """# Debug config - relaxed filter to test signal generation
name: keltner_2826_debug
data: SPY_5m_test  # Make sure this points to your test data

strategy:
  - keltner_bands:
      period: [30]
      multiplier: [1.0]
      
      # Try without filter first
      # filter: {volatility_above: {threshold: 1.1}}
      
      # Or try with lower threshold
      filter: {volatility_above: {threshold: 0.8}}
"""
    
    debug_path = Path(test_result_path).parent / "debug_config.yaml"
    with open(debug_path, 'w') as f:
        f.write(debug_config)
    
    print(f"Created debug config at: {debug_path}")
    print("\nTry running with the debug config to see if signals are generated")

if __name__ == "__main__":
    analyze_test_results()