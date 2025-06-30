"""Debug fill metadata to understand inverted stop losses."""

import pandas as pd
from pathlib import Path

# Load fills data
results_path = Path("config/bollinger/results/latest")
fills_path = results_path / "traces/execution/fills/execution_fills.parquet"

fills_df = pd.read_parquet(fills_path)

print("=== ANALYZING FILL METADATA ===\n")

# Find one inverted stop loss case
for idx, row in fills_df.iterrows():
    metadata = row['metadata']
    if isinstance(metadata, dict):
        nested = metadata.get('metadata', {})
        if isinstance(nested, dict) and nested.get('exit_type') == 'stop_loss':
            # Find the previous entry
            for j in range(idx-1, -1, -1):
                prev_row = fills_df.iloc[j]
                prev_meta = prev_row['metadata']
                if isinstance(prev_meta, dict):
                    prev_nested = prev_meta.get('metadata', {})
                    if not prev_nested.get('exit_type') and prev_meta.get('side') != metadata.get('side'):
                        entry_price = float(prev_meta.get('price', 0))
                        exit_price = float(metadata.get('price', 0))
                        if entry_price > 0:
                            return_pct = (exit_price - entry_price) / entry_price * 100
                            
                            # Check if it's inverted (positive instead of negative)
                            if abs(return_pct - 0.075) < 0.01:  # ~0.075% positive
                                print("FOUND INVERTED STOP LOSS:")
                                print(f"Entry price: ${entry_price:.4f}")
                                print(f"Exit price: ${exit_price:.4f}")
                                print(f"Return: {return_pct:.4f}% (should be -0.075%)")
                                print(f"\nExit fill metadata:")
                                
                                # Print all metadata
                                for key, value in metadata.items():
                                    print(f"  {key}: {value}")
                                
                                print(f"\nNested metadata:")
                                for key, value in nested.items():
                                    print(f"  {key}: {value}")
                                
                                # Calculate what the stop price should have been
                                correct_stop_price = entry_price * 0.99925  # -0.075%
                                print(f"\nCorrect stop price should be: ${correct_stop_price:.4f}")
                                print(f"Actual exit price: ${exit_price:.4f}")
                                print(f"Difference: ${exit_price - correct_stop_price:.4f}")
                                
                                # Check if exit price is entry + 0.075% (inverted)
                                inverted_price = entry_price * 1.00075
                                print(f"\nInverted price (entry + 0.075%): ${inverted_price:.4f}")
                                print(f"Matches actual exit? {abs(exit_price - inverted_price) < 0.01}")
                                
                                # Stop after first example
                                exit()

print("No inverted stop losses found!")