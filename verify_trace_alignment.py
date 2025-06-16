#!/usr/bin/env python3
"""Verify indexed sparse storage and alignment with source data."""

import json
import os
from pathlib import Path
import subprocess

def analyze_parquet_with_duckdb(parquet_path, limit=10):
    """Use DuckDB to analyze parquet file structure."""
    # Create SQL query to examine the parquet file
    sql_query = f"""
    SELECT * FROM '{parquet_path}' LIMIT {limit}
    """
    
    # Also get schema
    schema_query = f"""
    DESCRIBE SELECT * FROM '{parquet_path}'
    """
    
    # Get row count
    count_query = f"""
    SELECT COUNT(*) as total_rows FROM '{parquet_path}'
    """
    
    # Get column stats
    stats_query = f"""
    SELECT 
        MIN(idx) as min_idx,
        MAX(idx) as max_idx,
        COUNT(DISTINCT idx) as unique_indices,
        COUNT(*) as total_rows
    FROM '{parquet_path}'
    """
    
    print(f"\n=== Analyzing: {Path(parquet_path).name} ===")
    
    # Execute queries using duckdb CLI
    for label, query in [("Schema", schema_query), ("Row Count", count_query), 
                         ("Index Stats", stats_query), ("Sample Rows", sql_query)]:
        print(f"\n{label}:")
        try:
            result = subprocess.run(
                ['duckdb', '-c', query],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"Failed to run query: {e}")

def check_sparse_storage_patterns():
    """Check if the storage is truly sparse by analyzing gaps in indices."""
    
    # SQL to check for gaps in indices
    gap_check_query = """
    WITH indexed_data AS (
        SELECT 
            idx,
            LAG(idx) OVER (ORDER BY idx) as prev_idx,
            idx - LAG(idx) OVER (ORDER BY idx) as gap_size
        FROM '{}'
    )
    SELECT 
        COUNT(*) as total_changes,
        AVG(gap_size) as avg_gap,
        MAX(gap_size) as max_gap,
        SUM(CASE WHEN gap_size > 1 THEN 1 ELSE 0 END) as gaps_count
    FROM indexed_data
    WHERE gap_size IS NOT NULL
    """
    
    return gap_check_query

def analyze_source_data():
    """Analyze the source SPY_1m.parquet file."""
    source_file = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"
    
    if Path(source_file).exists():
        print("\n=== SOURCE DATA ANALYSIS ===")
        
        # Get row count and date range
        query = """
        SELECT 
            COUNT(*) as total_bars,
            MIN(datetime) as start_date,
            MAX(datetime) as end_date
        FROM '{}'
        """.format(source_file)
        
        try:
            result = subprocess.run(['duckdb', '-c', query], capture_output=True, text=True)
            if result.returncode == 0:
                print("Source data info:")
                print(result.stdout)
        except Exception as e:
            print(f"Error analyzing source: {e}")

def main():
    # Sample file paths
    signal_files = [
        "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_3fabc3f9/traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet",
        "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_3fabc3f9/traces/SPY_1m/signals/rsi_bands_grid/SPY_rsi_bands_grid_11_25_85.parquet",
    ]
    
    classifier_file = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_3fabc3f9/traces/SPY_1m/classifiers/market_regime_grid/SPY_market_regime_grid_0002_05.parquet"
    
    # First analyze source data
    analyze_source_data()
    
    # Analyze signal files
    print("\n\n=== SIGNAL FILES ANALYSIS ===")
    for file_path in signal_files:
        if Path(file_path).exists():
            analyze_parquet_with_duckdb(file_path, limit=5)
            
            # Check sparse patterns
            print("\nChecking sparse storage patterns:")
            gap_query = check_sparse_storage_patterns().format(file_path)
            result = subprocess.run(['duckdb', '-c', gap_query], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
    
    # Analyze classifier file
    print("\n\n=== CLASSIFIER FILE ANALYSIS ===")
    if Path(classifier_file).exists():
        analyze_parquet_with_duckdb(classifier_file, limit=5)
        
        # Check sparse patterns
        print("\nChecking sparse storage patterns:")
        gap_query = check_sparse_storage_patterns().format(classifier_file)
        result = subprocess.run(['duckdb', '-c', gap_query], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
    
    # Verify alignment by checking if indices from trace files exist in source data
    print("\n\n=== VERIFYING INDEX ALIGNMENT ===")
    
    # Get some sample indices from a trace file
    sample_query = """
    SELECT idx FROM '{}' LIMIT 5
    """.format(signal_files[0])
    
    result = subprocess.run(['duckdb', '-c', sample_query], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Sample indices from trace file:")
        print(result.stdout)
        
        # Now verify these indices correspond to valid rows in source data
        verify_query = """
        WITH trace_indices AS (
            SELECT DISTINCT idx FROM '{}' LIMIT 10
        ),
        source_data AS (
            SELECT 
                ROW_NUMBER() OVER (ORDER BY datetime) - 1 as row_idx,
                datetime,
                close
            FROM '{}'
        )
        SELECT 
            t.idx as trace_idx,
            s.datetime,
            s.close
        FROM trace_indices t
        JOIN source_data s ON t.idx = s.row_idx
        ORDER BY t.idx
        """.format(signal_files[0], "/Users/daws/ADMF-PC/data/SPY_1m.parquet")
        
        print("\nVerifying that trace indices map to source data rows:")
        result = subprocess.run(['duckdb', '-c', verify_query], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Verification failed: {result.stderr}")

if __name__ == "__main__":
    main()