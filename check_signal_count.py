#!/usr/bin/env python3
"""
Check how many strategies actually generated signals in the latest workspace.
"""

import duckdb

# Connect to the analytics database
db_path = 'workspaces/20250614_211925_indicator_grid_v3_SPY/analytics.duckdb'
conn = duckdb.connect(db_path, read_only=True)

print("=== Strategy Signal Analysis ===\n")

# Check total strategies configured
total_strategies = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
print(f"Total strategies configured: {total_strategies}")

# Check if there are any signals at all
try:
    # First check what tables exist
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"Available tables: {[t[0] for t in tables]}")
    
    # Try to find signals
    signal_tables = [t[0] for t in tables if 'signal' in t[0].lower() or 'trace' in t[0].lower()]
    print(f"Signal-related tables: {signal_tables}")
    
    # Check if there's an event_archives table (common in ADMF-PC)
    if 'event_archives' in [t[0] for t in tables]:
        print("\n=== Checking event_archives ===")
        event_count = conn.execute("SELECT COUNT(*) FROM event_archives").fetchone()[0]
        print(f"Total events in archives: {event_count}")
        
        if event_count > 0:
            # Check event types
            event_types = conn.execute("""
                SELECT event_type, COUNT(*) as count
                FROM event_archives 
                GROUP BY event_type 
                ORDER BY count DESC
            """).fetchall()
            
            print("\nEvent types:")
            for event_type, count in event_types:
                print(f"  - {event_type}: {count}")
            
            # Check for SIGNAL events specifically
            signal_events = conn.execute("""
                SELECT COUNT(*) 
                FROM event_archives 
                WHERE event_type = 'SIGNAL'
            """).fetchone()[0]
            
            print(f"\nSIGNAL events: {signal_events}")
            
            if signal_events > 0:
                # Get unique strategies that generated signals
                unique_strategies = conn.execute("""
                    SELECT COUNT(DISTINCT strategy_id)
                    FROM event_archives 
                    WHERE event_type = 'SIGNAL'
                """).fetchone()[0]
                
                print(f"Unique strategies with signals: {unique_strategies}")
                
                # Show breakdown by strategy
                strategy_counts = conn.execute("""
                    SELECT strategy_id, COUNT(*) as signal_count
                    FROM event_archives 
                    WHERE event_type = 'SIGNAL'
                    GROUP BY strategy_id
                    ORDER BY signal_count DESC
                    LIMIT 20
                """).fetchall()
                
                print(f"\nTop 20 strategies by signal count:")
                for strategy_id, count in strategy_counts:
                    print(f"  - {strategy_id}: {count} signals")
                
                # Check for strategies with no signals
                no_signal_strategies = conn.execute("""
                    SELECT s.strategy_id, s.strategy_type
                    FROM strategies s
                    LEFT JOIN (
                        SELECT DISTINCT strategy_id 
                        FROM event_archives 
                        WHERE event_type = 'SIGNAL'
                    ) e ON s.strategy_id = e.strategy_id
                    WHERE e.strategy_id IS NULL
                    ORDER BY s.strategy_type, s.strategy_id
                """).fetchall()
                
                print(f"\nStrategies with NO signals ({len(no_signal_strategies)}):")
                
                # Group by type
                from collections import defaultdict
                by_type = defaultdict(list)
                for strategy_id, strategy_type in no_signal_strategies:
                    by_type[strategy_type].append(strategy_id)
                
                for strategy_type, strategies in sorted(by_type.items()):
                    print(f"  {strategy_type} ({len(strategies)}):")
                    for strategy in strategies[:5]:  # Show first 5
                        print(f"    - {strategy}")
                    if len(strategies) > 5:
                        print(f"    ... and {len(strategies) - 5} more")
                
    else:
        print("No event_archives table found")
        
except Exception as e:
    print(f"Error checking signals: {e}")

# Summary
print(f"\n=== SUMMARY ===")
if 'signal_events' in locals():
    signaling_strategies = unique_strategies if 'unique_strategies' in locals() else 0
    non_signaling = total_strategies - signaling_strategies
    
    print(f"Total strategies: {total_strategies}")
    print(f"Strategies with signals: {signaling_strategies} ({signaling_strategies/total_strategies*100:.1f}%)")
    print(f"Strategies without signals: {non_signaling} ({non_signaling/total_strategies*100:.1f}%)")
    print(f"Total signals generated: {signal_events}")
else:
    print(f"Could not determine signal counts")

conn.close()