
-- Step 1: Get exact time period
WITH time_analysis AS (
    SELECT 
        MIN(ts) as start_time,
        MAX(ts) as end_time,
        -- Calculate exact days
        (EXTRACT('days' FROM (CAST('2025-01-17' AS DATE) - CAST('2024-03-26' AS DATE))) + 1) as total_days
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
),

-- Step 2: Count actual signals for one strategy to verify
macd_signal_count AS (
    SELECT 
        COUNT(*) as total_signal_changes
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
),

-- Step 3: Get regime distribution with exact counts
regime_distribution AS (
    SELECT 
        val as regime,
        COUNT(*) as regime_minutes,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as regime_pct
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    GROUP BY val
)

-- Show verification data
SELECT 'TIME_PERIOD' as analysis, 
       CAST(total_days AS VARCHAR) as detail,
       CAST(total_days / 365.25 AS VARCHAR) as value
FROM time_analysis

UNION ALL

SELECT 'SIGNAL_COUNT' as analysis,
       'macd_total_signals' as detail, 
       CAST(total_signal_changes AS VARCHAR) as value
FROM macd_signal_count

UNION ALL

SELECT 'REGIME_DIST' as analysis,
       regime as detail,
       CAST(regime_minutes AS VARCHAR) as value
FROM regime_distribution

ORDER BY analysis, detail;
