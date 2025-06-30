#!/usr/bin/env python3
"""
Analyze SPY workspace using DuckDB directly
"""
import subprocess
from pathlib import Path

def analyze_duckdb_workspace(workspace_path):
    """Analyze workspace using DuckDB CLI"""
    
    workspace = Path(workspace_path)
    analytics_db = workspace / "analytics.duckdb"
    
    if not analytics_db.exists():
        print(f"Error: No analytics.duckdb found in {workspace_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing DuckDB Analytics: {analytics_db}")
    print(f"{'='*60}\n")
    
    # Series of queries to explore the database
    queries = [
        # 1. List all tables
        ("Tables in Database", "SHOW TABLES;"),
        
        # 2. Check strategies table
        ("Strategy Count", "SELECT COUNT(*) as total_strategies FROM strategies;"),
        
        # 3. Strategy types
        ("Strategy Types", """
            SELECT strategy_type, COUNT(*) as count 
            FROM strategies 
            GROUP BY strategy_type 
            ORDER BY count DESC;
        """),
        
        # 4. Sample strategies
        ("Sample Strategies (Top 10)", """
            SELECT strategy_id, strategy_name, strategy_type, signal_file_path
            FROM strategies
            LIMIT 10;
        """),
        
        # 5. Check for performance data
        ("Check for Performance Columns", "DESCRIBE strategies;"),
        
        # 6. Look for runs table
        ("Run Information", """
            SELECT COUNT(*) as total_runs FROM runs;
        """),
        
        # 7. Check classifiers
        ("Classifier Information", """
            SELECT COUNT(*) as total_classifiers FROM classifiers;
        """),
        
        # 8. Sample classifier data
        ("Sample Classifiers", """
            SELECT classifier_id, classifier_name, classifier_type
            FROM classifiers
            LIMIT 5;
        """),
    ]
    
    for title, query in queries:
        print(f"\n{title}:")
        print("-" * len(title))
        
        result = subprocess.run(
            ['duckdb', str(analytics_db), '-csv'],
            input=query,
            text=True,
            capture_output=True
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                # Format output nicely
                lines = output.split('\n')
                for line in lines:
                    print(f"  {line}")
            else:
                print("  (No results)")
        else:
            # Try without the table if it doesn't exist
            if "does not exist" in result.stderr:
                print("  (Table does not exist)")
            else:
                print(f"  Error: {result.stderr.strip()}")
    
    # Try to find signal files referenced in the database
    print(f"\n\nChecking for Signal Files:")
    print("-" * 25)
    
    signal_query = """
        SELECT DISTINCT signal_file_path 
        FROM strategies 
        WHERE signal_file_path IS NOT NULL 
        LIMIT 20;
    """
    
    result = subprocess.run(
        ['duckdb', str(analytics_db), '-csv', '-noheader'],
        input=signal_query,
        text=True,
        capture_output=True
    )
    
    if result.returncode == 0 and result.stdout.strip():
        signal_paths = result.stdout.strip().split('\n')
        existing_count = 0
        
        for signal_path in signal_paths[:5]:  # Check first 5
            full_path = workspace / signal_path
            exists = full_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {signal_path}")
            if exists:
                existing_count += 1
        
        if len(signal_paths) > 5:
            print(f"  ... and {len(signal_paths) - 5} more")
        
        print(f"\n  Found {existing_count} existing signal files out of {len(signal_paths)} checked")
    
    print(f"\n{'='*60}\n")


def main():
    workspace_path = "workspaces/20250617_194112_signal_generation_SPY"
    
    # Convert to absolute path
    workspace_abs = Path(workspace_path).absolute()
    
    if not workspace_abs.exists():
        # Try from project root
        project_root = Path(__file__).parent
        workspace_abs = project_root / workspace_path
    
    if workspace_abs.exists():
        analyze_duckdb_workspace(str(workspace_abs))
    else:
        print(f"Error: Could not find workspace at {workspace_path}")


if __name__ == "__main__":
    main()