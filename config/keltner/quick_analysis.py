#!/usr/bin/env python3
"""
Quick analysis of Keltner strategy results.
Run this from the config/keltner directory.
"""

import sys
sys.path.append('../../src')

from analytics.simple_analytics import TraceAnalysis
import pandas as pd
import json

# Load the data
print("Loading trace data...")
ta = TraceAnalysis('results/20250622_155944')
print(ta)
print()

# 1. Overview
print("=== OVERVIEW ===")
overview = ta.sql("""
    SELECT 
        COUNT(DISTINCT strategy_id) as num_strategies,
        COUNT(*) as total_signals,
        MIN(idx) as first_bar,
        MAX(idx) as last_bar
    FROM traces
""")
print(overview)
print()

# 2. Signal Activity
print("=== SIGNAL ACTIVITY ===")
activity = ta.sql("""
    SELECT 
        strategy_id,
        COUNT(*) as num_signals,
        SUM(CASE WHEN val > 0 THEN 1 ELSE 0 END) as long_signals,
        SUM(CASE WHEN val < 0 THEN 1 ELSE 0 END) as short_signals
    FROM traces
    GROUP BY strategy_id
    ORDER BY num_signals DESC
    LIMIT 10
""")
print("Top 10 most active strategies:")
print(activity)
print()

# 3. Extract trades and calculate performance
print("=== PERFORMANCE ANALYSIS ===")
print("Extracting trades from signals...")

trades = ta.sql("""
    WITH signal_changes AS (
        SELECT 
            strategy_id,
            idx as bar_idx,
            val as signal_value,
            px as price,
            LAG(val, 1, 0) OVER (PARTITION BY strategy_id ORDER BY idx) as prev_signal
        FROM traces
    ),
    entries AS (
        SELECT 
            strategy_id,
            bar_idx as entry_bar,
            price as entry_price,
            signal_value as entry_signal,
            ROW_NUMBER() OVER (PARTITION BY strategy_id ORDER BY bar_idx) as entry_num
        FROM signal_changes
        WHERE signal_value != 0 AND prev_signal = 0
    ),
    exits AS (
        SELECT 
            strategy_id,
            bar_idx as exit_bar,
            price as exit_price,
            ROW_NUMBER() OVER (PARTITION BY strategy_id ORDER BY bar_idx) as exit_num
        FROM signal_changes
        WHERE signal_value = 0 AND prev_signal != 0
    )
    SELECT 
        e.strategy_id,
        COUNT(*) as num_trades,
        AVG(
            CASE 
                WHEN e.entry_signal > 0 THEN (x.exit_price - e.entry_price) / e.entry_price
                ELSE (e.entry_price - x.exit_price) / e.entry_price
            END
        ) as avg_return,
        STDDEV(
            CASE 
                WHEN e.entry_signal > 0 THEN (x.exit_price - e.entry_price) / e.entry_price
                ELSE (e.entry_price - x.exit_price) / e.entry_price
            END
        ) as return_std
    FROM entries e
    LEFT JOIN exits x 
        ON e.strategy_id = x.strategy_id 
        AND x.exit_num = e.entry_num
    WHERE x.exit_bar IS NOT NULL
    GROUP BY e.strategy_id
    HAVING COUNT(*) > 10
""")

# Calculate Sharpe ratio
trades['sharpe'] = trades['avg_return'] / trades['return_std'] * (252 ** 0.5)  # Annualized

# Show top performers
print("\nTop 10 strategies by Sharpe ratio (min 10 trades):")
top_performers = trades.sort_values('sharpe', ascending=False).head(10)
print(top_performers)
print()

# 4. Production candidates
print("=== PRODUCTION CANDIDATES ===")
production = trades[
    (trades['sharpe'] > 1.5) & 
    (trades['num_trades'] > 50)
].sort_values('sharpe', ascending=False)

print(f"Found {len(production)} production-ready strategies")
print("Criteria: Sharpe > 1.5, Trades > 50")
print()

if len(production) > 0:
    print("Best 5 strategies:")
    print(production.head())
    
    # Save for later use
    production.head().to_csv('best_strategies.csv')
    print("\nSaved to best_strategies.csv")

# 5. Signal distribution analysis
print("\n=== SIGNAL DISTRIBUTION ===")
signal_dist = ta.sql("""
    SELECT 
        CASE 
            WHEN num_signals < 50 THEN '<50'
            WHEN num_signals < 100 THEN '50-100'
            WHEN num_signals < 150 THEN '100-150'
            WHEN num_signals < 200 THEN '150-200'
            ELSE '>200'
        END as signal_range,
        COUNT(*) as strategy_count
    FROM (
        SELECT strategy_id, COUNT(*) as num_signals
        FROM traces
        GROUP BY strategy_id
    )
    GROUP BY signal_range
    ORDER BY signal_range
""")
print(signal_dist)

print("\n=== DONE ===")
print("For interactive analysis, open analysis.ipynb in Jupyter")