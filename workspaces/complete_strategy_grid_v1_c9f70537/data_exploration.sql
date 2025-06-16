
-- Explore what px field contains and find source data
WITH sample_signals AS (
    SELECT ts, val, px, strat 
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
    ORDER BY ts
    LIMIT 10
),

px_analysis AS (
    SELECT 
        MIN(px) as min_px,
        MAX(px) as max_px,
        COUNT(DISTINCT px) as unique_px_values,
        COUNT(*) as total_signals
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
)

-- Show sample data
SELECT 'SAMPLE_SIGNALS' as analysis, ts as detail, CAST(val AS VARCHAR) as value, CAST(px AS VARCHAR) as extra
FROM sample_signals

UNION ALL

SELECT 'PX_ANALYSIS' as analysis, 'min_px' as detail, CAST(min_px AS VARCHAR) as value, NULL as extra
FROM px_analysis

UNION ALL

SELECT 'PX_ANALYSIS' as analysis, 'max_px' as detail, CAST(max_px AS VARCHAR) as value, NULL as extra  
FROM px_analysis

UNION ALL

SELECT 'PX_ANALYSIS' as analysis, 'unique_values' as detail, CAST(unique_px_values AS VARCHAR) as value, NULL as extra
FROM px_analysis

ORDER BY analysis, detail;
