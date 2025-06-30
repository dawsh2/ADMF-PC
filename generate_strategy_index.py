#!/usr/bin/env python3
"""
Generate strategy index for the bollinger run
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fix_notebook_strategy_index import create_strategy_index_from_metadata, save_strategy_index

run_dir = "config/bollinger/results/20250623_062931"
print(f"Generating strategy index for: {run_dir}")

# Create the index
strategy_index = create_strategy_index_from_metadata(run_dir)

if strategy_index is not None:
    # Save it
    output_path = save_strategy_index(run_dir, strategy_index)
    print(f"\nStrategy index saved to: {output_path}")
    print(f"\nTotal strategies: {len(strategy_index)}")
    print(f"\nSample of parameters:")
    print(strategy_index[['strategy_hash', 'param_period', 'param_std_dev', 'param_exit_threshold']].head(10))
else:
    print("Failed to create strategy index")