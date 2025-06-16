
-- Check for zero values in different strategies
WITH signal_distributions AS (
    SELECT 'MACD_CROSSOVER' as strategy, 
           CAST(val AS INTEGER) as signal_val, 
           COUNT(*) as count
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
    GROUP BY val
    
    UNION ALL
    
    SELECT 'WILLIAMS_R' as strategy,
           CAST(val AS INTEGER) as signal_val,
           COUNT(*) as count
    FROM read_parquet('traces/SPY_1m/signals/williams_r_grid/SPY_williams_r_grid_7_-80_-20.parquet')
    GROUP BY val
    
    UNION ALL
    
    SELECT 'RSI_THRESHOLD' as strategy,
           CAST(val AS INTEGER) as signal_val,
           COUNT(*) as count
    FROM read_parquet('traces/SPY_1m/signals/rsi_threshold_grid/SPY_rsi_threshold_grid_11_50.parquet')
    GROUP BY val
    
    UNION ALL
    
    SELECT 'CCI_BANDS' as strategy,
           CAST(val AS INTEGER) as signal_val,
           COUNT(*) as count
    FROM read_parquet('traces/SPY_1m/signals/cci_bands_grid/SPY_cci_bands_grid_11_-100_100.parquet')
    GROUP BY val
)

SELECT 
    strategy,
    signal_val,
    count,
    ROUND(count * 100.0 / SUM(count) OVER (PARTITION BY strategy), 1) as percentage
FROM signal_distributions
ORDER BY strategy, signal_val;
