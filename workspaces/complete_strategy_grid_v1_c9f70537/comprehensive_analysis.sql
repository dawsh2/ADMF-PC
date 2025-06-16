
-- Load classifier data with regime mapping
WITH classifier_regimes AS (
    SELECT 
        ts,
        val as regime_name
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
),

-- Get regime time distribution  
regime_stats AS (
    SELECT 
        regime_name,
        COUNT(*) as points,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct
    FROM classifier_regimes
    GROUP BY regime_name
),

-- Load strategy signals with all values
strategy_signals AS (
    SELECT 
        ts,
        val as signal_val,
        px as price,
        ROW_NUMBER() OVER (ORDER BY ts) as signal_order
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
    ORDER BY ts
),

-- Calculate time period
time_period AS (
    SELECT 
        MIN(ts) as start_time,
        MAX(ts) as end_time,
        EXTRACT('days' FROM (MAX(ts) - MIN(ts))) / 365.25 as total_years
    FROM classifier_regimes
)

-- Show comprehensive analysis
SELECT 'REGIME_DISTRIBUTION' as analysis, regime_name as detail, 
       points as count, pct as percentage
FROM regime_stats

UNION ALL

SELECT 'TIME_PERIOD' as analysis, 'total_years' as detail,
       total_years as count, NULL as percentage
FROM time_period

UNION ALL  

SELECT 'SIGNAL_DISTRIBUTION' as analysis, CAST(signal_val AS VARCHAR) as detail,
       COUNT(*) as count, COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
FROM strategy_signals
GROUP BY signal_val

UNION ALL

SELECT 'TOTAL_SIGNALS' as analysis, 'all_signal_changes' as detail,
       COUNT(*) as count, NULL as percentage
FROM strategy_signals

ORDER BY analysis, count DESC;
