"""Check if filtered signals were generated properly"""
import duckdb
import pandas as pd

# Connect to analytics database
conn = duckdb.connect("workspaces/signal_generation_310b2aeb/analytics.duckdb")

# Check what tables exist
print("=== Tables in database ===")
tables = conn.execute("SHOW TABLES").fetchall()
for table in tables:
    print(f"  {table[0]}")
print()

# Check strategies table structure
print("=== Strategies table structure ===")
strategies_schema = conn.execute("DESCRIBE strategies").fetchdf()
print(strategies_schema)

# Check strategies table
print("\n=== Strategies in database ===")
strategies_query = """
SELECT strategy_name, strategy_type, parameters 
FROM strategies 
ORDER BY strategy_name
LIMIT 10
"""
strategies = conn.execute(strategies_query).fetchdf()
print(strategies)
print(f"\nTotal strategies: {conn.execute('SELECT COUNT(*) FROM strategies').fetchone()[0]}")

# Check if parameters contain filters
print("\n=== Parameters with Filters ===")
filter_check_query = """
SELECT strategy_name, parameters
FROM strategies
WHERE parameters LIKE '%filter%'
LIMIT 10
"""
try:
    filtered_params = conn.execute(filter_check_query).fetchdf()
    if len(filtered_params) > 0:
        print(f"Found {len(filtered_params)} strategies with filters:")
        for _, row in filtered_params.iterrows():
            print(f"\n{row['strategy_name']}:")
            print(f"  {row['parameters']}")
    else:
        print("No strategies with filters found")
except Exception as e:
    print(f"Error checking filters: {e}")

# Check sparse signal storage
print("\n=== Sparse Signal Storage ===")
sparse_query = """
SELECT strategy_id, COUNT(*) as change_count
FROM sparse_signal_changes
GROUP BY strategy_id
ORDER BY change_count DESC
LIMIT 10
"""
try:
    sparse_counts = conn.execute(sparse_query).fetchdf()
    print(sparse_counts)
    
    # Get total number of strategies with filters
    total_with_filters = conn.execute("""
    SELECT COUNT(*) 
    FROM strategies 
    WHERE parameters LIKE '%filter%'
    """).fetchone()[0]
    print(f"\nTotal strategies with filters: {total_with_filters}")
    
except Exception as e:
    print(f"Error: {e}")

conn.close()