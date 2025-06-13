#!/usr/bin/env python3
"""
Simple DuckDB SQL interface
Usage: python duckdb_cli.py [database_path] [optional_sql_query]
"""
import duckdb
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python duckdb_cli.py <database_path> [sql_query]")
        print("Example: python duckdb_cli.py analytics.duckdb")
        print("Example: python duckdb_cli.py analytics.duckdb 'SELECT COUNT(*) FROM signal_changes'")
        return
    
    db_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        # Run single query
        query = " ".join(sys.argv[2:])
        conn = duckdb.connect(db_path)
        try:
            result = conn.execute(query).fetchdf()
            print(result.to_string(index=False))
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()
    else:
        # Interactive mode
        conn = duckdb.connect(db_path)
        print(f"Connected to {db_path}")
        print("Enter SQL queries (type 'quit' or Ctrl+D to exit):")
        
        try:
            while True:
                query = input("duckdb> ").strip()
                if query.lower() in ['quit', 'exit']:
                    break
                if not query:
                    continue
                
                try:
                    if query.upper().startswith('SELECT') or query.upper().startswith('SHOW') or query.upper().startswith('DESCRIBE'):
                        result = conn.execute(query).fetchdf()
                        print(result.to_string(index=False))
                    else:
                        conn.execute(query)
                        print("Query executed successfully")
                except Exception as e:
                    print(f"Error: {e}")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
        finally:
            conn.close()

if __name__ == "__main__":
    main()