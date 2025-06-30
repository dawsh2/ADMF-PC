"""Analyze the Keltner Bands workspace signal_generation_a3628f6c"""
import duckdb
import pandas as pd
from pathlib import Path

workspace = Path("workspaces/signal_generation_a3628f6c")
db_path = workspace / "analytics.duckdb"

print("=== ANALYZING KELTNER BANDS WORKSPACE ===")
print(f"Workspace: {workspace}")
print(f"Database: {db_path}")

# Connect to DuckDB
conn = duckdb.connect(str(db_path), read_only=True)

# Check what tables exist
print("\n\nAvailable tables:")
tables = conn.execute("SHOW TABLES").fetchall()
for table in tables:
    print(f"  - {table[0]}")
    
# Check for traces or signals
if tables:
    # Try to find signal-related tables
    for table_name in [t[0] for t in tables]:
        if 'signal' in table_name.lower() or 'trace' in table_name.lower():
            print(f"\n\nChecking table: {table_name}")
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  Row count: {count}")
            
            if count > 0:
                # Show sample data
                print("\n  Sample data:")
                sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchdf()
                print(sample)
                
                # Get schema
                print("\n  Schema:")
                schema = conn.execute(f"DESCRIBE {table_name}").fetchdf()
                print(schema)

# Check for performance/trade data
for potential_table in ['trades', 'performance', 'results', 'analysis']:
    try:
        df = conn.execute(f"SELECT * FROM {potential_table}").fetchdf()
        if len(df) > 0:
            print(f"\n\nFound {potential_table} table with {len(df)} rows")
            print(df.head())
    except:
        pass

# Check metadata
try:
    metadata = conn.execute("SELECT * FROM metadata").fetchdf()
    print("\n\nMetadata:")
    print(metadata)
except:
    pass

# Check if there's a summary table
try:
    summary = conn.execute("SELECT * FROM summary").fetchdf()
    print("\n\nSummary:")
    print(summary)
except:
    pass

conn.close()

print("\n\nCONCLUSION:")
print("This workspace was testing Keltner Bands strategy.")
print("The metadata shows 0 signals, suggesting the strategy didn't generate trades.")
print("This could be due to:")
print("1. Keltner Bands parameters not suitable for the data")
print("2. Strategy implementation issues")
print("3. Data period too short (only January 2024)")
print("4. The strategy might be stored differently than swing_pivot_bounce")