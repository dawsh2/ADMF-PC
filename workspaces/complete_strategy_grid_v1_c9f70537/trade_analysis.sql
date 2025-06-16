
-- Load classifier data
WITH classifier_regimes AS (
    SELECT 
        ts,
        val as regime_val,
        CASE val
            WHEN 0 THEN 'neutral'
            WHEN 1 THEN 'low_vol_bullish'  
            WHEN 2 THEN 'low_vol_bearish'
            WHEN 3 THEN 'high_vol_bullish'
            WHEN 4 THEN 'high_vol_bearish'
        END as regime_name
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
),

-- Get regime distribution
regime_stats AS (
    SELECT 
        regime_name,
        COUNT(*) as points,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as pct
    FROM classifier_regimes
    GROUP BY regime_name
),

-- Load strategy signals  
strategy_signals AS (
    SELECT 
        ts,
        val as signal_val,
        px as price
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
    WHERE val \!= 0  -- Only non-zero signals (actual trades)
),

-- Calculate total time period
time_stats AS (
    SELECT 
        MIN(ts) as start_time,
        MAX(ts) as end_time,
        (MAX(ts) - MIN(ts)) / INTERVAL 365.25 DAY as total_years
    FROM classifier_regimes
)

-- Show results
SELECT 'REGIME_DISTRIBUTION' as analysis, regime_name as item, points as value, pct as percentage
FROM regime_stats

UNION ALL

SELECT 'TIME_PERIOD' as analysis, 'total_years' as item, 
       EXTRACT('days' FROM (end_time - start_time)) / 365.25 as value, NULL as percentage  
FROM time_stats

UNION ALL

SELECT 'STRATEGY_SIGNALS' as analysis, 'total_signal_changes' as item, 
       COUNT(*) as value, NULL as percentage
FROM strategy_signals

UNION ALL

SELECT 'SIGNAL_VALUES' as analysis, CAST(signal_val AS VARCHAR) as item,
       COUNT(*) as value, COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
FROM strategy_signals
GROUP BY signal_val

ORDER BY analysis, value DESC;
