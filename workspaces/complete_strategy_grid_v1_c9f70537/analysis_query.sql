
-- Load one volatility momentum classifier for regime mapping
WITH classifier_data AS (
    SELECT 
        ts,
        CASE val
            WHEN 0 THEN 'neutral'
            WHEN 1 THEN 'low_vol_bullish'
            WHEN 2 THEN 'low_vol_bearish' 
            WHEN 3 THEN 'high_vol_bullish'
            WHEN 4 THEN 'high_vol_bearish'
        END as regime
    FROM 'traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_16_80_30.parquet'
),

-- Calculate regime time distributions
regime_stats AS (
    SELECT 
        regime,
        COUNT(*) as regime_count,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as regime_pct
    FROM classifier_data
    GROUP BY regime
),

-- Sample one high-frequency strategy 
strategy_signals AS (
    SELECT 
        ts,
        val as signal_val,
        px as price
    FROM 'traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_35_9.parquet'
    ORDER BY ts
)

-- Show regime distributions and sample strategy info
SELECT 
    'REGIME_STATS' as analysis_type,
    regime as item,
    regime_count as count,
    regime_pct as percentage,
    NULL as signal_info
FROM regime_stats

UNION ALL

SELECT 
    'STRATEGY_INFO' as analysis_type,
    'total_signals' as item,
    COUNT(*) as count,
    NULL as percentage,
    'Sample: SPY_macd_crossover_grid_5_35_9.parquet' as signal_info
FROM strategy_signals

UNION ALL

SELECT 
    'SIGNAL_VALUES' as analysis_type,
    CAST(signal_val AS VARCHAR) as item,
    COUNT(*) as count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage,
    NULL as signal_info
FROM strategy_signals
GROUP BY signal_val

ORDER BY analysis_type, count DESC;
