#!/usr/bin/env python3
"""
Simple analysis of latest Keltner traces
"""

import sys
sys.path.append('../../src')
from analytics.simple_analytics import TraceAnalysis

# Load the latest data
print("Loading latest trace data...")
ta = TraceAnalysis('results/20250622_180858')
print(f"Loaded: {ta}\n")

# 1. How many strategies do we have?
print("=== BASIC STATS ===")
total = ta.sql("SELECT COUNT(DISTINCT strategy_id) as count FROM traces")
print(f"Total strategies: {total['count'][0]}")

# 2. How many signals total?
signals = ta.sql("SELECT COUNT(*) as count FROM traces")
print(f"Total signals: {signals['count'][0]}")

# 3. Sample of the data
print("\n=== SAMPLE DATA ===")
sample = ta.sql("SELECT * FROM traces LIMIT 5")
print(sample)

# 4. Most active strategy
print("\n=== MOST ACTIVE STRATEGY ===")
active = ta.sql("""
    SELECT strategy_id, COUNT(*) as signals 
    FROM traces 
    GROUP BY strategy_id 
    ORDER BY signals DESC 
    LIMIT 1
""")
print(active)

# 5. Calculate simple performance with trades per day
print("\n=== SIMPLE PERFORMANCE (Top 10) ===")
perf = ta.sql("""
    WITH signal_changes AS (
        SELECT 
            strategy_id,
            bar_idx,
            signal_value,
            price,
            LAG(signal_value) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as prev_signal
        FROM traces
    ),
    entries AS (
        SELECT 
            strategy_id,
            bar_idx as entry_bar,
            price as entry_price,
            signal_value as direction
        FROM signal_changes
        WHERE signal_value != 0 AND (prev_signal = 0 OR prev_signal IS NULL)
    ),
    exits AS (
        SELECT 
            strategy_id,
            bar_idx as exit_bar,
            price as exit_price
        FROM signal_changes
        WHERE signal_value = 0 AND prev_signal != 0
    ),
    completed_trades AS (
        SELECT 
            e.strategy_id,
            e.entry_bar,
            e.entry_price,
            e.direction,
            x.exit_bar,
            x.exit_price,
            (x.exit_price - e.entry_price) / e.entry_price * e.direction * 100 as return_pct
        FROM entries e
        JOIN exits x ON e.strategy_id = x.strategy_id AND x.exit_bar > e.entry_bar
        WHERE x.exit_bar = (
            SELECT MIN(exit_bar) 
            FROM exits 
            WHERE strategy_id = e.strategy_id AND exit_bar > e.entry_bar
        )
    ),
    strategy_stats AS (
        SELECT 
            strategy_id,
            COUNT(*) as completed_trades,
            MIN(entry_bar) as first_bar,
            MAX(exit_bar) as last_bar,
            EXP(SUM(LN(1 + return_pct/100))) * 100 - 100 as total_return_pct,
            AVG(return_pct) as avg_return_pct
        FROM completed_trades
        GROUP BY strategy_id
    )
    SELECT 
        s.strategy_id,
        s.completed_trades,
        ROUND((s.last_bar - s.first_bar) / 78.0, 1) as trading_days,
        ROUND(s.completed_trades / ((s.last_bar - s.first_bar) / 78.0), 2) as trades_per_day,
        ROUND(s.avg_return_pct, 4) as avg_return_pct,
        ROUND(s.total_return_pct, 2) as total_return_pct,
        ROUND(POWER(1 + s.total_return_pct/100, 252.0/((s.last_bar - s.first_bar) / 78.0)) * 100 - 100, 2) as annualized_return_pct
    FROM strategy_stats s
    WHERE s.completed_trades > 10 AND (s.last_bar - s.first_bar) > 78
    ORDER BY s.total_return_pct DESC
    LIMIT 10
""")
print(perf)

print("\n=== DONE ===")
print("To run more queries, use:")
print('  result = ta.sql("YOUR SQL HERE")')
print('  print(result)')