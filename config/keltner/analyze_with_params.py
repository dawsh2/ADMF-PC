#!/usr/bin/env python3
"""
Analyze strategies WITH their parameters visible
"""

import sys
sys.path.append('../../src')
from analytics.simple_analytics import TraceAnalysis
import pandas as pd
import json

# Load data
ta = TraceAnalysis('results/20250622_180858')

# First, create the parameter mapping
import yaml
from itertools import product

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

keltner_config = config['strategy'][0]['keltner_bands']
periods = keltner_config['period']
multipliers = keltner_config['multiplier']
filters = keltner_config['filter']

# Create strategy parameter table in DuckDB
params_data = []
strategy_id = 0

for period in periods:
    for multiplier in multipliers:
        for filter_idx, filter_config in enumerate(filters):
            params_data.append({
                'strategy_id': strategy_id,
                'period': period,
                'multiplier': multiplier,
                'filter_index': filter_idx,
                'has_filter': filter_config is not None
            })
            strategy_id += 1

# Create params DataFrame and load into DuckDB
params_df = pd.DataFrame(params_data)
ta.con.register('strategy_params', params_df)

print("=== TOP PERFORMERS WITH PARAMETERS ===")
result = ta.sql("""
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
    strategy_performance AS (
        SELECT 
            strategy_id,
            COUNT(*) as trades,
            EXP(SUM(LN(1 + return_pct/100))) * 100 - 100 as total_return_pct,
            AVG(return_pct) as avg_return_pct
        FROM completed_trades
        GROUP BY strategy_id
    )
    SELECT 
        p.strategy_id,
        p.trades,
        ROUND(p.total_return_pct, 2) as total_return_pct,
        ROUND(p.avg_return_pct, 4) as avg_return_pct,
        sp.period,
        sp.multiplier,
        sp.filter_index,
        sp.has_filter
    FROM strategy_performance p
    JOIN strategy_params sp ON p.strategy_id = sp.strategy_id
    WHERE p.trades > 50
    ORDER BY p.total_return_pct DESC
    LIMIT 20
""")

print(result)

# Specifically show strategy 1029
print("\n=== STRATEGY 1029 DETAILS ===")
s1029 = ta.sql("""
    SELECT * FROM strategy_params WHERE strategy_id = 1029
""")
print(s1029)

# Find similar strategies to 1029
print("\n=== STRATEGIES SIMILAR TO 1029 ===")
if len(s1029) > 0:
    period = s1029['period'][0]
    multiplier = s1029['multiplier'][0]
    
    similar = ta.sql(f"""
        WITH performance AS (
            -- Same performance calc as above
            SELECT strategy_id, COUNT(*) as trades
            FROM traces
            GROUP BY strategy_id
        )
        SELECT 
            sp.strategy_id,
            sp.period,
            sp.multiplier,
            sp.filter_index,
            p.trades
        FROM strategy_params sp
        LEFT JOIN performance p ON sp.strategy_id = p.strategy_id
        WHERE sp.period = {period} AND sp.multiplier = {multiplier}
        ORDER BY sp.filter_index
        LIMIT 10
    """)
    print(similar)