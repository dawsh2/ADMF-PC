#!/usr/bin/env python3
"""Check what types of exits are happening."""

import pandas as pd
import json
from pathlib import Path

print("=== Checking Exit Types ===")

# Check latest results
results_dir = Path("config/bollinger/results/latest")
closes_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"

if closes_file.exists():
    closes = pd.read_parquet(closes_file)
    
    # Parse metadata
    if 'metadata' in closes.columns:
        for i in range(len(closes)):
            if isinstance(closes.iloc[i]['metadata'], str):
                try:
                    meta = json.loads(closes.iloc[i]['metadata'])
                    for key, value in meta.items():
                        if key not in closes.columns:
                            closes.loc[closes.index[i], key] = value
                except:
                    pass
    
    print(f"Total position closes: {len(closes)}")
    
    # Count exit types
    if 'exit_type' in closes.columns:
        print("\nExit type distribution:")
        exit_counts = closes['exit_type'].value_counts()
        for exit_type, count in exit_counts.items():
            pct = count / len(closes) * 100
            print(f"  {exit_type}: {count} ({pct:.1f}%)")
        
        # Check if trailing stops are dominating
        if 'trailing_stop' in exit_counts:
            print(f"\n⚠️ Trailing stops: {exit_counts.get('trailing_stop', 0)}")
            print("This might be preventing regular stop losses from triggering")
    
    # Check average returns by exit type
    if 'exit_type' in closes.columns and 'entry_price' in closes.columns and 'exit_price' in closes.columns:
        print("\nAverage return by exit type:")
        for exit_type in closes['exit_type'].unique():
            mask = closes['exit_type'] == exit_type
            subset = closes[mask]
            if len(subset) > 0:
                # Simple return calculation (not adjusted for short/long)
                returns = (subset['exit_price'] - subset['entry_price']) / subset['entry_price'] * 100
                avg_return = returns.mean()
                print(f"  {exit_type}: {avg_return:.4f}%")
                
                if exit_type == 'trailing_stop':
                    print("    Trailing stop is 0.001% (very tight!)")

else:
    print("No position close data found")