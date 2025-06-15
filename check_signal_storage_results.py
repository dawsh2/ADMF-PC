#!/usr/bin/env python3
"""
Check signal storage results from the grid search run
"""
import duckdb
import json

# Connect to the most recent analytics database
db_path = "workspaces/expansive_grid_search_5fe966d1/analytics.duckdb"
conn = duckdb.connect(db_path, read_only=True)

# Check what tables exist
print("Tables in database:")
tables = conn.execute("SHOW TABLES").fetchall()
for table in tables:
    print(f"  - {table[0]}")

# Check signals table
print("\n=== SIGNALS TABLE ===")
signal_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
print(f"Total signals stored: {signal_count:,}")

# Get unique strategies
unique_strategies = conn.execute("""
    SELECT COUNT(DISTINCT strategy_id) as count
    FROM signals
""").fetchone()[0]
print(f"Unique strategies that generated signals: {unique_strategies}")

# Get signal distribution by strategy
print("\nSignals per strategy type:")
strategy_signals = conn.execute("""
    SELECT 
        SPLIT_PART(strategy_id, '_', 1) as strategy_type,
        COUNT(DISTINCT strategy_id) as unique_configs,
        COUNT(*) as total_signals,
        COUNT(DISTINCT timestamp) as unique_timestamps
    FROM signals
    GROUP BY SPLIT_PART(strategy_id, '_', 1)
    ORDER BY COUNT(*) DESC
    LIMIT 20
""").fetchall()

for row in strategy_signals:
    print(f"  {row[0]}: {row[1]} configs, {row[2]:,} signals across {row[3]} timestamps")

# Check non-zero signals
print("\nNon-zero signal distribution:")
non_zero_signals = conn.execute("""
    SELECT 
        signal_value,
        COUNT(*) as count,
        COUNT(DISTINCT strategy_id) as strategies
    FROM signals
    WHERE signal_value != 0
    GROUP BY signal_value
    ORDER BY signal_value
""").fetchall()

for row in non_zero_signals:
    print(f"  Signal {row[0]}: {row[1]:,} occurrences from {row[2]} strategies")

# Check classifications
print("\n=== CLASSIFICATIONS TABLE ===")
if 'classifications' in [t[0] for t in tables]:
    class_count = conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
    print(f"Total classifications stored: {class_count:,}")
    
    # Get unique classifiers
    unique_classifiers = conn.execute("""
        SELECT COUNT(DISTINCT classifier_id) as count
        FROM classifications
    """).fetchone()[0]
    print(f"Unique classifiers that generated classifications: {unique_classifiers}")
else:
    print("No classifications table found")

# Check metadata
print("\n=== METADATA ===")
with open("workspaces/expansive_grid_search_5fe966d1/metadata.json", "r") as f:
    metadata = json.load(f)
    print(f"Run ID: {metadata.get('run_id', 'N/A')}")
    print(f"Symbol: {metadata.get('symbol', 'N/A')}")
    print(f"Total bars processed: {metadata.get('total_bars', 'N/A')}")
    
    if 'strategies' in metadata:
        print(f"Total strategies configured: {len(metadata['strategies'])}")
    
    if 'classifiers' in metadata:
        print(f"Total classifiers configured: {len(metadata['classifiers'])}")

# Get sample of strategies that didn't generate signals
print("\n=== STRATEGIES WITHOUT SIGNALS ===")
all_strategy_ids = conn.execute("""
    SELECT DISTINCT json_extract_string(value, '$.strategy_id') as strategy_id
    FROM (
        SELECT json_array_elements(strategies::json) as value
        FROM (SELECT :strategies as strategies)
    )
""", {"strategies": json.dumps(metadata.get('strategies', []))}).fetchall()

strategies_with_signals = conn.execute("""
    SELECT DISTINCT strategy_id FROM signals
""").fetchall()

strategies_with_signals_set = set(s[0] for s in strategies_with_signals)
all_strategies_set = set(s[0] for s in all_strategy_ids if s[0])

strategies_without_signals = all_strategies_set - strategies_with_signals_set
print(f"Strategies without any signals: {len(strategies_without_signals)} out of {len(all_strategies_set)}")

if strategies_without_signals:
    print("Sample of strategies without signals:")
    for i, strat in enumerate(list(strategies_without_signals)[:10]):
        print(f"  - {strat}")

conn.close()