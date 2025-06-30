#!/usr/bin/env python3
"""Analyze the bollinger_10 test results"""

import json
import os
import pandas as pd
from pathlib import Path

# Path to bollinger_10 results
base_path = Path("config/bollinger/bollinger_10/results/20250623_070236")

print("=== Bollinger Strategy #10 Test Results ===")
print(f"Configuration: period=11, std_dev=2.0")
print(f"Data: SPY_5m")
print()

# Read metadata
metadata_path = base_path / "metadata.json"
if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("Metadata Summary:")
    print(f"- Workflow ID: {metadata['workflow_id']}")
    print(f"- Total bars processed: {metadata['total_bars']:,}")
    print(f"- Total signals generated: {metadata['total_signals']}")
    print(f"- Total classifications: {metadata['total_classifications']}")
    print(f"- Stored changes: {metadata['stored_changes']}")
    print()
    
    if metadata['total_signals'] == 0:
        print("⚠️ WARNING: No signals were generated!")
        print("This means the strategy did not produce any buy/sell signals during the test period.")
        print()
        print("Possible reasons:")
        print("1. The parameters (period=11, std_dev=2.0) might be too conservative")
        print("2. The market conditions during the test period didn't trigger any signals")
        print("3. There might be an issue with the strategy implementation or configuration")
else:
    print("ERROR: Could not find metadata.json")

# Check for any trace files
traces_path = base_path / "traces"
if traces_path.exists():
    trace_files = list(traces_path.glob("**/*.parquet"))
    print(f"\nTrace files found: {len(trace_files)}")
    if len(trace_files) == 0:
        print("No trace files were generated (consistent with 0 signals)")

# Let's also check if we can find results from the larger bollinger test
print("\n=== Checking parent bollinger test for strategy #10 ===")
parent_traces = Path("config/bollinger/results/20250623_062931/traces/bollinger_bands")
strategy_10_file = parent_traces / "SPY_5m_compiled_strategy_10.parquet"

if strategy_10_file.exists():
    print(f"Found strategy #10 results in parent test: {strategy_10_file}")
    try:
        df = pd.read_parquet(strategy_10_file)
        print(f"- Trace file has {len(df)} records")
        if len(df) > 0:
            print("- This suggests the strategy CAN generate signals with different data or timeframe")
    except Exception as e:
        print(f"- Could not read parquet file: {e}")
else:
    print("Could not find strategy #10 in parent test results")

print("\n=== Recommendations ===")
print("1. The bollinger_10 test ran successfully but generated no trading signals")
print("2. This is likely due to conservative parameters (period=11, std_dev=2.0)")
print("3. Consider testing with:")
print("   - Smaller standard deviation (e.g., 1.5 or 1.0) for more signals")
print("   - Different time periods to see if market conditions affect signal generation")
print("   - Adding debug output to understand why signals aren't being generated")