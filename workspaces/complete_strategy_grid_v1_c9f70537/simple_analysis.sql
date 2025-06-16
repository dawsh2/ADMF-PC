
-- Basic data inspection
WITH strategy_data AS (
    SELECT 
        ts,
        CAST(val AS INTEGER) as signal_val,
        px as price
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
    ORDER BY ts
),

classifier_data AS (
    SELECT 
        ts,
        val as regime
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    ORDER BY ts
)

-- Show signal value distribution
SELECT 
    'SIGNAL_VALUES' as analysis_type,
    signal_val,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM strategy_data
GROUP BY signal_val

UNION ALL

-- Show regime distribution  
SELECT 
    'REGIME_VALUES' as analysis_type,
    regime as signal_val,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM classifier_data
GROUP BY regime

UNION ALL

-- Show totals
SELECT 
    'TOTALS' as analysis_type,
    'strategy_signals' as signal_val,
    COUNT(*) as count,
    NULL as percentage
FROM strategy_data

UNION ALL

SELECT 
    'TOTALS' as analysis_type, 
    'classifier_points' as signal_val,
    COUNT(*) as count,
    NULL as percentage
FROM classifier_data

ORDER BY analysis_type, signal_val;
