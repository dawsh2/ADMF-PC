#!/usr/bin/env python3
"""
Simple SQL test for analytics data
"""
import duckdb
import sys

def run_sql_command(query):
    """Run a single SQL command and display results"""
    try:
        conn = duckdb.connect('analytics.duckdb')
        result = conn.execute(query).fetchdf()
        print(f"\nQuery: {query}")
        print("=" * 50)
        print(result.to_string(index=False))
        print(f"\nRows returned: {len(result)}")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific query from command line
        query = " ".join(sys.argv[1:])
        run_sql_command(query)
    else:
        # Run some test queries
        print("🔍 Testing Analytics Database")
        
        # Show tables
        run_sql_command("SHOW TABLES")
        
        # Show signal summary
        run_sql_command("""
            SELECT 
                component_type,
                strategy_type,
                COUNT(*) as configurations,
                AVG(signal_frequency) as avg_signal_freq
            FROM component_metrics 
            GROUP BY component_type, strategy_type
            ORDER BY avg_signal_freq DESC
        """)