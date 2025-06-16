#!/usr/bin/env python3
"""Check the schema of source data file."""

import subprocess
from pathlib import Path

def check_source_schema():
    source_file = "/Users/daws/ADMF-PC/data/SPY_1m.parquet"
    
    if Path(source_file).exists():
        # Get schema
        schema_query = f"DESCRIBE SELECT * FROM '{source_file}'"
        
        print("=== SOURCE DATA SCHEMA ===")
        result = subprocess.run(['duckdb', '-c', schema_query], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        
        # Get sample rows
        sample_query = f"SELECT * FROM '{source_file}' LIMIT 5"
        
        print("\n=== SOURCE DATA SAMPLE ===")
        result = subprocess.run(['duckdb', '-c', sample_query], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            
        # Check if there's a natural row order
        print("\n=== ROW NUMBERING TEST ===")
        row_number_query = f"""
        SELECT 
            ROW_NUMBER() OVER (ORDER BY timestamp) - 1 as computed_idx,
            timestamp,
            close
        FROM '{source_file}'
        LIMIT 10
        """
        
        result = subprocess.run(['duckdb', '-c', row_number_query], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)

if __name__ == "__main__":
    check_source_schema()