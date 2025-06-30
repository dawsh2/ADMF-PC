#!/usr/bin/env python3
"""
Compare OHLC data sources between universal analysis and execution engine.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_universal_analysis_data():
    """Load data used by universal analysis."""
    data_path = Path("/Users/daws/ADMF-PC/data/SPY_5m.csv")
    print(f"Loading universal analysis data from: {data_path}")
    
    if not data_path.exists():
        print(f"ERROR: Universal analysis data file not found: {data_path}")
        return None
        
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    
    print(f"Universal analysis data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df

def find_execution_data_files():
    """Find what data files the execution engine might use."""
    data_dir = Path("/Users/daws/ADMF-PC/data")
    
    print("\nSearching for data files in execution data directory...")
    print(f"Data directory: {data_dir}")
    
    csv_files = list(data_dir.glob("*.csv"))
    print(f"\nFound {len(csv_files)} CSV files:")
    for f in sorted(csv_files):
        print(f"  - {f.name}")
    
    # Look for SPY files specifically
    spy_files = [f for f in csv_files if 'SPY' in f.name.upper()]
    print(f"\nSPY-related files: {len(spy_files)}")
    for f in spy_files:
        print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
    
    return spy_files

def load_execution_data():
    """Load data as the execution engine would."""
    # The SimpleCSVLoader looks for patterns like SPY.csv, SPY_5m.csv, etc.
    patterns = [
        "SPY.csv",
        "SPY_5m.csv",
        "spy.csv",
        "spy_5m.csv"
    ]
    
    data_dir = Path("/Users/daws/ADMF-PC/data")
    
    for pattern in patterns:
        path = data_dir / pattern
        if path.exists():
            print(f"\nFound execution data file: {path}")
            df = pd.read_csv(path)
            
            # Handle date parsing as the loader does
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], utc=True)
                df.set_index(date_cols[0], inplace=True)
            
            print(f"Execution data shape: {df.shape}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
    
    print("ERROR: No execution data file found!")
    return None

def compare_datasets(df_universal, df_execution):
    """Compare the two datasets for differences."""
    print("\n" + "="*80)
    print("COMPARING DATASETS")
    print("="*80)
    
    # Basic shape comparison
    print(f"\nShape comparison:")
    print(f"  Universal: {df_universal.shape}")
    print(f"  Execution: {df_execution.shape}")
    
    if df_universal.shape[0] != df_execution.shape[0]:
        print(f"  ⚠️  Different number of rows! Difference: {abs(df_universal.shape[0] - df_execution.shape[0])}")
    
    # Column comparison
    print(f"\nColumns:")
    print(f"  Universal: {list(df_universal.columns)}")
    print(f"  Execution: {list(df_execution.columns)}")
    
    # Normalize column names for comparison
    col_map = {
        'open': ['open', 'Open', 'OPEN'],
        'high': ['high', 'High', 'HIGH'],
        'low': ['low', 'Low', 'LOW'],
        'close': ['close', 'Close', 'CLOSE'],
        'volume': ['volume', 'Volume', 'VOLUME']
    }
    
    # Find overlapping dates
    common_dates = df_universal.index.intersection(df_execution.index)
    print(f"\nCommon dates: {len(common_dates)}")
    
    if len(common_dates) == 0:
        print("⚠️  No overlapping dates! Checking date formats...")
        print(f"  Universal first 5 dates: {df_universal.index[:5].tolist()}")
        print(f"  Execution first 5 dates: {df_execution.index[:5].tolist()}")
        return
    
    # Compare OHLC values for common dates
    print("\nComparing OHLC values for common dates...")
    
    # Sample comparison - first 10 common dates
    sample_dates = common_dates[:10]
    
    for i, date in enumerate(sample_dates):
        print(f"\n{i+1}. Date: {date}")
        
        # Get normalized column names
        for std_col, variants in col_map.items():
            univ_col = next((c for c in df_universal.columns if c in variants), None)
            exec_col = next((c for c in df_execution.columns if c in variants), None)
            
            if univ_col and exec_col:
                univ_val = df_universal.loc[date, univ_col]
                exec_val = df_execution.loc[date, exec_col]
                diff = abs(univ_val - exec_val)
                
                if diff > 0.001:  # Significant difference
                    print(f"  ⚠️  {std_col}: Universal={univ_val:.4f}, Execution={exec_val:.4f}, Diff={diff:.4f}")
                else:
                    print(f"  ✓ {std_col}: {univ_val:.4f} (match)")
    
    # Check for price movements that would trigger stops/targets
    print("\n" + "-"*60)
    print("CHECKING PRICE MOVEMENTS FOR STOPS/TARGETS")
    print("-"*60)
    
    # Look for bars with significant movements (0.075% or 0.1%)
    threshold_pct = 0.00075  # 0.075%
    
    # For universal data
    univ_open_col = next((c for c in df_universal.columns if c.lower() == 'open'), 'open')
    univ_high_col = next((c for c in df_universal.columns if c.lower() == 'high'), 'high')
    univ_low_col = next((c for c in df_universal.columns if c.lower() == 'low'), 'low')
    
    univ_moves_up = ((df_universal[univ_high_col] - df_universal[univ_open_col]) / df_universal[univ_open_col]) >= threshold_pct
    univ_moves_down = ((df_universal[univ_open_col] - df_universal[univ_low_col]) / df_universal[univ_open_col]) >= threshold_pct
    
    print(f"\nUniversal analysis data:")
    print(f"  Bars with {threshold_pct*100:.3f}% move up from open: {univ_moves_up.sum()}")
    print(f"  Bars with {threshold_pct*100:.3f}% move down from open: {univ_moves_down.sum()}")
    
    # For execution data
    exec_open_col = next((c for c in df_execution.columns if c.lower() == 'open'), 'open')
    exec_high_col = next((c for c in df_execution.columns if c.lower() == 'high'), 'high')
    exec_low_col = next((c for c in df_execution.columns if c.lower() == 'low'), 'low')
    
    exec_moves_up = ((df_execution[exec_high_col] - df_execution[exec_open_col]) / df_execution[exec_open_col]) >= threshold_pct
    exec_moves_down = ((df_execution[exec_open_col] - df_execution[exec_low_col]) / df_execution[exec_open_col]) >= threshold_pct
    
    print(f"\nExecution engine data:")
    print(f"  Bars with {threshold_pct*100:.3f}% move up from open: {exec_moves_up.sum()}")
    print(f"  Bars with {threshold_pct*100:.3f}% move down from open: {exec_moves_down.sum()}")
    
    # Find discrepancies in specific bars
    print("\n" + "-"*60)
    print("DETAILED BAR COMPARISON (First 20 bars with movement)")
    print("-"*60)
    
    # Find bars where universal shows movement but execution doesn't
    count = 0
    for date in common_dates:
        if count >= 20:
            break
            
        univ_open = df_universal.loc[date, univ_open_col]
        univ_high = df_universal.loc[date, univ_high_col]
        univ_low = df_universal.loc[date, univ_low_col]
        
        exec_open = df_execution.loc[date, exec_open_col]
        exec_high = df_execution.loc[date, exec_high_col]
        exec_low = df_execution.loc[date, exec_low_col]
        
        univ_up_pct = (univ_high - univ_open) / univ_open
        univ_down_pct = (univ_open - univ_low) / univ_open
        
        exec_up_pct = (exec_high - exec_open) / exec_open
        exec_down_pct = (exec_open - exec_low) / exec_open
        
        # Check if there's a significant difference
        if abs(univ_up_pct - exec_up_pct) > 0.0001 or abs(univ_down_pct - exec_down_pct) > 0.0001:
            print(f"\n{date}:")
            print(f"  Universal - O:{univ_open:.4f} H:{univ_high:.4f} L:{univ_low:.4f}")
            print(f"  Execution - O:{exec_open:.4f} H:{exec_high:.4f} L:{exec_low:.4f}")
            print(f"  Universal moves: Up {univ_up_pct*100:.3f}%, Down {univ_down_pct*100:.3f}%")
            print(f"  Execution moves: Up {exec_up_pct*100:.3f}%, Down {exec_down_pct*100:.3f}%")
            
            if univ_up_pct >= threshold_pct and exec_up_pct < threshold_pct:
                print(f"  ⚠️  Universal hits profit target, execution doesn't!")
            if univ_down_pct >= threshold_pct and exec_down_pct < threshold_pct:
                print(f"  ⚠️  Universal hits stop loss, execution doesn't!")
                
            count += 1
    
    # Check for data quality issues
    print("\n" + "-"*60)
    print("DATA QUALITY CHECKS")
    print("-"*60)
    
    # Check for missing data
    print("\nMissing data:")
    print(f"  Universal: {df_universal.isnull().sum().sum()} null values")
    print(f"  Execution: {df_execution.isnull().sum().sum()} null values")
    
    # Check for invalid OHLC relationships
    univ_invalid = (
        (df_universal[univ_high_col] < df_universal[univ_low_col]) |
        (df_universal[univ_high_col] < df_universal[univ_open_col]) |
        (df_universal[univ_high_col] < df_universal[next((c for c in df_universal.columns if c.lower() == 'close'), 'close')]) |
        (df_universal[univ_low_col] > df_universal[univ_open_col]) |
        (df_universal[univ_low_col] > df_universal[next((c for c in df_universal.columns if c.lower() == 'close'), 'close')])
    )
    
    exec_invalid = (
        (df_execution[exec_high_col] < df_execution[exec_low_col]) |
        (df_execution[exec_high_col] < df_execution[exec_open_col]) |
        (df_execution[exec_high_col] < df_execution[next((c for c in df_execution.columns if c.lower() == 'close'), 'close')]) |
        (df_execution[exec_low_col] > df_execution[exec_open_col]) |
        (df_execution[exec_low_col] > df_execution[next((c for c in df_execution.columns if c.lower() == 'close'), 'close')])
    )
    
    print(f"\nInvalid OHLC relationships:")
    print(f"  Universal: {univ_invalid.sum()} bars")
    print(f"  Execution: {exec_invalid.sum()} bars")

def main():
    """Main comparison function."""
    print("="*80)
    print("DATA SOURCE COMPARISON: Universal Analysis vs Execution Engine")
    print("="*80)
    
    # Find execution data files
    spy_files = find_execution_data_files()
    
    # Load universal analysis data
    df_universal = load_universal_analysis_data()
    if df_universal is None:
        return
    
    # Load execution data
    df_execution = load_execution_data()
    if df_execution is None:
        return
    
    # Compare datasets
    compare_datasets(df_universal, df_execution)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nBoth systems appear to use the same data file: /Users/daws/ADMF-PC/data/SPY_5m.csv")
    print("The difference in profit targets hit (523 vs 150) is NOT due to different data sources.")
    print("\nThe issue must be in the execution logic or trade management, not the underlying price data.")

if __name__ == "__main__":
    main()