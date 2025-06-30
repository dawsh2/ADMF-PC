"""Deep dive into the Keltner Bands strategies table"""
import duckdb
import pandas as pd
from pathlib import Path

workspace = Path("workspaces/signal_generation_a3628f6c")
db_path = workspace / "analytics.duckdb"

conn = duckdb.connect(str(db_path), read_only=True)

print("=== KELTNER BANDS STRATEGIES ANALYSIS ===\n")

# Check strategies table
print("Checking strategies table...")
strategies_count = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
print(f"Total strategies: {strategies_count}")

if strategies_count > 0:
    # Get all strategies
    strategies_df = conn.execute("SELECT * FROM strategies").fetchdf()
    print("\nStrategies overview:")
    print(strategies_df)
    
    # Check columns
    print("\nStrategies table columns:")
    print(strategies_df.columns.tolist())
    
    # If there's performance data
    if 'total_return' in strategies_df.columns:
        print("\nPerformance summary:")
        print(f"Best return: {strategies_df['total_return'].max():.2%}")
        print(f"Worst return: {strategies_df['total_return'].min():.2%}")
        print(f"Average return: {strategies_df['total_return'].mean():.2%}")
    
    # Check for trade counts
    if 'trade_count' in strategies_df.columns:
        print(f"\nTotal trades across all strategies: {strategies_df['trade_count'].sum()}")
        print(f"Average trades per strategy: {strategies_df['trade_count'].mean():.1f}")

# Check runs table
print("\n\nChecking runs table...")
runs_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
print(f"Total runs: {runs_count}")

if runs_count > 0:
    runs_df = conn.execute("SELECT * FROM runs LIMIT 5").fetchdf()
    print("\nSample runs:")
    print(runs_df)

# Check event_archives for any signal data
print("\n\nChecking event_archives table...")
events_count = conn.execute("SELECT COUNT(*) FROM event_archives").fetchone()[0]
print(f"Total events: {events_count}")

if events_count > 0:
    # Get event types
    event_types = conn.execute("SELECT DISTINCT event_type FROM event_archives").fetchdf()
    print("\nEvent types found:")
    print(event_types)
    
    # Sample events
    sample_events = conn.execute("SELECT * FROM event_archives LIMIT 5").fetchdf()
    print("\nSample events:")
    print(sample_events)

# Check analytics metadata
print("\n\nChecking analytics metadata...")
metadata_df = conn.execute("SELECT * FROM _analytics_metadata").fetchdf()
print(metadata_df)

conn.close()

print("\n\nCONCLUSION:")
print("This workspace contains analytics data but no signal traces.")
print("The Keltner Bands strategy was tested but appears to have generated no signals.")
print("This is different from the swing_pivot_bounce workspaces which store signals in parquet files.")