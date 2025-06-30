import sys
sys.path.append('../../src')
from analytics.simple_analytics import TraceAnalysis

# Load the latest results
ta = TraceAnalysis('results/20250622_180858')

# Basic overview
print('=== Overview ===')
overview = ta.sql('''
    SELECT 
        COUNT(DISTINCT strategy_id) as total_strategies,
        COUNT(*) as total_signals,
        MIN(bar_idx) as first_bar,
        MAX(bar_idx) as last_bar,
        COUNT(DISTINCT strategy_id || '_' || bar_idx) as unique_events
    FROM traces
''')
print(overview)

print('\n=== Signal Distribution ===')
signals = ta.sql('''
    SELECT 
        signal_value,
        COUNT(*) as count,
        COUNT(DISTINCT strategy_id) as strategies_using
    FROM traces
    GROUP BY signal_value
    ORDER BY signal_value
''')
print(signals)

print('\n=== Most Active Strategies (Top 10) ===')
active = ta.sql('''
    SELECT 
        strategy_id,
        COUNT(*) as signal_changes,
        MIN(bar_idx) as first_signal,
        MAX(bar_idx) as last_signal,
        COUNT(DISTINCT signal_value) as unique_signals
    FROM traces
    GROUP BY strategy_id
    ORDER BY signal_changes DESC
    LIMIT 10
''')
print(active)

print('\n=== Least Active Strategies (Top 10) ===')
inactive = ta.sql('''
    SELECT 
        strategy_id,
        COUNT(*) as signal_changes,
        MIN(bar_idx) as first_signal,
        MAX(bar_idx) as last_signal
    FROM traces
    GROUP BY strategy_id
    ORDER BY signal_changes ASC
    LIMIT 10
''')
print(inactive)

print('\n=== Trade Extraction Example (Strategy 0) ===')
trades = ta.sql('''
    WITH signal_changes AS (
        SELECT 
            strategy_id,
            bar_idx,
            signal_value,
            price,
            LAG(signal_value) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as prev_signal
        FROM traces
        WHERE strategy_id = 0
    ),
    entries AS (
        SELECT 
            bar_idx as entry_bar,
            price as entry_price,
            signal_value as direction
        FROM signal_changes
        WHERE signal_value != 0 AND (prev_signal = 0 OR prev_signal IS NULL)
    ),
    exits AS (
        SELECT 
            bar_idx as exit_bar,
            price as exit_price,
            LAG(signal_value) OVER (ORDER BY bar_idx) as direction
        FROM signal_changes
        WHERE signal_value = 0 AND prev_signal != 0
    )
    SELECT 
        e.entry_bar,
        e.entry_price,
        x.exit_bar,
        x.exit_price,
        e.direction,
        (x.exit_price - e.entry_price) / e.entry_price * e.direction * 100 as return_pct
    FROM entries e
    JOIN exits x ON x.exit_bar > e.entry_bar
    WHERE x.exit_bar = (
        SELECT MIN(exit_bar) FROM exits WHERE exit_bar > e.entry_bar
    )
    LIMIT 5
''')
print(trades)

# Save query for calculating performance metrics
print('\n=== Performance Calculation Query ===')
print("""
To calculate performance for all strategies, use:

ta.sql('''
    WITH signal_changes AS (
        SELECT 
            strategy_id,
            bar_idx,
            signal_value,
            price,
            LAG(signal_value) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as prev_signal
        FROM traces
    ),
    trades AS (
        SELECT 
            strategy_id,
            bar_idx as entry_bar,
            price as entry_price,
            signal_value as direction,
            LEAD(bar_idx) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as exit_bar,
            LEAD(price) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as exit_price
        FROM signal_changes
        WHERE signal_value != 0 AND (prev_signal = 0 OR prev_signal IS NULL)
    )
    SELECT 
        strategy_id,
        COUNT(*) as num_trades,
        AVG((exit_price - entry_price) / entry_price * direction * 100) as avg_return_pct,
        SUM((exit_price - entry_price) / entry_price * direction * 100) as total_return_pct
    FROM trades
    WHERE exit_bar IS NOT NULL
    GROUP BY strategy_id
    HAVING COUNT(*) > 10
    ORDER BY avg_return_pct DESC
    LIMIT 20
''')
""")