
-- Analysis across multiple high-frequency strategies
WITH strategy_files AS (
    SELECT * FROM (VALUES
        ('macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet', 'macd_crossover_5_20_11'),
        ('macd_crossover_grid/SPY_macd_crossover_grid_12_26_9.parquet', 'macd_crossover_12_26_9'),
        ('williams_r_grid/SPY_williams_r_grid_7_-80_-20.parquet', 'williams_r_7_-80_-20'),
        ('williams_r_grid/SPY_williams_r_grid_14_-85_-15.parquet', 'williams_r_14_-85_-15'),
        ('rsi_threshold_grid/SPY_rsi_threshold_grid_7_50.parquet', 'rsi_threshold_7_50'),
        ('rsi_threshold_grid/SPY_rsi_threshold_grid_11_45.parquet', 'rsi_threshold_11_45'),
        ('cci_bands_grid/SPY_cci_bands_grid_11_-100_100.parquet', 'cci_bands_11_-100_100'),
        ('cci_bands_grid/SPY_cci_bands_grid_19_-80_80.parquet', 'cci_bands_19_-80_80')
    ) AS t(file_path, strategy_name)
),

-- Load classifier for regime mapping
classifier_regimes AS (
    SELECT 
        ts,
        val as regime
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
),

-- Calculate total time period
time_stats AS (
    SELECT 
        COUNT(*) as total_minutes,
        COUNT(*) / (252.0 * 390.0) as total_years  -- Approximate trading year
    FROM classifier_regimes
),

-- Get regime time distribution
regime_time_dist AS (
    SELECT 
        regime,
        COUNT(*) as regime_minutes,
        COUNT(*) / (252.0 * 390.0) as regime_years,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as regime_pct
    FROM classifier_regimes
    GROUP BY regime
)

-- Show basic stats first
SELECT 'TIME_ANALYSIS' as type, regime as detail, 
       regime_years as value, regime_pct as percentage
FROM regime_time_dist

UNION ALL

SELECT 'TOTAL_TIME' as type, 'total_years' as detail,
       total_years as value, NULL as percentage
FROM time_stats

ORDER BY type, value DESC;
