#!/usr/bin/env python3
"""
Query the analytics DuckDB database to find price data and understand available tables.
"""

import duckdb
from pathlib import Path

def explore_database(db_path: Path):
    """Explore the analytics database structure."""
    print(f"🔍 Exploring database: {db_path}")
    
    try:
        conn = duckdb.connect(str(db_path))
        
        # List all tables
        print("\n📋 Available tables:")
        tables = conn.execute("SHOW TABLES").fetchall()
        for table in tables:
            print(f"  - {table[0]}")
        
        # For each table, show structure and sample data
        for table in tables:
            table_name = table[0]
            print(f"\n{'='*60}")
            print(f"TABLE: {table_name}")
            print(f"{'='*60}")
            
            # Get table info
            try:
                info = conn.execute(f"DESCRIBE {table_name}").fetchall()
                print("Columns:")
                for col_info in info:
                    print(f"  {col_info[0]}: {col_info[1]}")
                
                # Get row count
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                print(f"\nRow count: {count:,}")
                
                # Show sample data
                if count > 0:
                    sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchall()
                    print(f"\nSample data (first 5 rows):")
                    for row in sample:
                        print(f"  {row}")
                
                # For tables with timestamps, show date range
                try:
                    ts_cols = [col[0] for col in info if 'timestamp' in col[0].lower() or 'ts' in col[0].lower() or 'time' in col[0].lower()]
                    if ts_cols:
                        ts_col = ts_cols[0]
                        date_range = conn.execute(f"SELECT MIN({ts_col}), MAX({ts_col}) FROM {table_name}").fetchone()
                        print(f"\nDate range ({ts_col}): {date_range[0]} to {date_range[1]}")
                except:
                    pass
                
            except Exception as e:
                print(f"Error examining table {table_name}: {e}")
        
        # Look for price/market data specifically
        print(f"\n{'='*60}")
        print("SEARCHING FOR PRICE DATA")
        print(f"{'='*60}")
        
        for table in tables:
            table_name = table[0]
            try:
                # Check if table has price-related columns
                columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
                price_cols = [col[0] for col in columns if any(term in col[0].lower() for term in ['price', 'px', 'close', 'open', 'high', 'low'])]
                
                if price_cols:
                    print(f"\n📈 Table '{table_name}' has price columns: {price_cols}")
                    
                    # Show sample of price data
                    price_sample = conn.execute(f"SELECT * FROM {table_name} WHERE {price_cols[0]} > 0 LIMIT 5").fetchall()
                    if price_sample:
                        print(f"Sample price data:")
                        for row in price_sample:
                            print(f"  {row}")
                    else:
                        print("No non-zero price data found")
            except:
                pass
        
        conn.close()
        
    except Exception as e:
        print(f"Error exploring database: {e}")

def main():
    """Main function."""
    workspace_path = Path("/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_9c2c22c9")
    db_path = workspace_path / "analytics.duckdb"
    
    explore_database(db_path)

if __name__ == "__main__":
    main()