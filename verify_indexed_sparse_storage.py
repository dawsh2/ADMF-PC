#!/usr/bin/env python3
"""Verify indexed sparse storage format in event trace files."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_trace_file(file_path):
    """Analyze a single trace file for indexed sparse storage format."""
    df = pd.read_parquet(file_path)
    
    analysis = {
        'file': file_path.name,
        'columns': list(df.columns),
        'shape': df.shape,
        'has_idx': 'idx' in df.columns,
        'has_required_cols': all(col in df.columns for col in ['idx', 'ts', 'sym', 'val', 'strat']),
    }
    
    if 'idx' in df.columns and len(df) > 0:
        # Check if idx values are monotonic (sorted)
        analysis['idx_monotonic'] = df['idx'].is_monotonic_increasing
        analysis['idx_min'] = int(df['idx'].min())
        analysis['idx_max'] = int(df['idx'].max())
        analysis['idx_range'] = int(df['idx'].max() - df['idx'].min())
        
        # Check for sparse storage (gaps in idx)
        if len(df) > 1:
            idx_diffs = df['idx'].diff().dropna()
            analysis['has_gaps'] = (idx_diffs > 1).any()
            analysis['avg_gap'] = float(idx_diffs.mean())
            analysis['max_gap'] = int(idx_diffs.max())
            
        # Check value changes
        if 'val' in df.columns:
            val_changes = df['val'].ne(df['val'].shift()).sum()
            analysis['value_changes'] = val_changes
            analysis['sparse_ratio'] = len(df) / (analysis['idx_range'] + 1) if analysis['idx_range'] > 0 else 1.0
            
    return analysis

def main():
    # Sample signal files
    signal_files = [
        "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_3fabc3f9/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet",
        "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_3fabc3f9/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet",
    ]
    
    # Sample classifier files
    classifier_files = [
        "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_3fabc3f9/traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_0002_05.parquet",
        "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_3fabc3f9/traces/SPY_1m/classifiers/microstructure_grid/SPY_microstructure_grid_00015_00003.parquet",
    ]
    
    print("=== SIGNAL FILES ANALYSIS ===")
    for file_path in signal_files:
        if Path(file_path).exists():
            analysis = analyze_trace_file(Path(file_path))
            print(f"\nFile: {analysis['file']}")
            print(f"Columns: {analysis['columns']}")
            print(f"Shape: {analysis['shape']}")
            print(f"Has idx column: {analysis['has_idx']}")
            print(f"Has required columns: {analysis['has_required_cols']}")
            if analysis.get('idx_monotonic') is not None:
                print(f"Index monotonic: {analysis['idx_monotonic']}")
                print(f"Index range: {analysis['idx_min']} to {analysis['idx_max']} ({analysis['idx_range']} bars)")
                print(f"Has gaps (sparse): {analysis.get('has_gaps', 'N/A')}")
                print(f"Average gap: {analysis.get('avg_gap', 'N/A'):.2f}")
                print(f"Sparse ratio: {analysis.get('sparse_ratio', 'N/A'):.4f}")
                
            # Show first few rows
            df = pd.read_parquet(Path(file_path))
            print("\nFirst 5 rows:")
            print(df.head())
            
    print("\n\n=== CLASSIFIER FILES ANALYSIS ===")
    for file_path in classifier_files:
        if Path(file_path).exists():
            analysis = analyze_trace_file(Path(file_path))
            print(f"\nFile: {analysis['file']}")
            print(f"Columns: {analysis['columns']}")
            print(f"Shape: {analysis['shape']}")
            print(f"Has idx column: {analysis['has_idx']}")
            print(f"Has required columns: {analysis['has_required_cols']}")
            if analysis.get('idx_monotonic') is not None:
                print(f"Index monotonic: {analysis['idx_monotonic']}")
                print(f"Index range: {analysis['idx_min']} to {analysis['idx_max']} ({analysis['idx_range']} bars)")
                print(f"Has gaps (sparse): {analysis.get('has_gaps', 'N/A')}")
                print(f"Average gap: {analysis.get('avg_gap', 'N/A'):.2f}")
                print(f"Sparse ratio: {analysis.get('sparse_ratio', 'N/A'):.4f}")
                
            # Show first few rows
            df = pd.read_parquet(Path(file_path))
            print("\nFirst 5 rows:")
            print(df.head())

if __name__ == "__main__":
    main()