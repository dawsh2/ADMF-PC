
-- Get actual time period from data
WITH time_check AS (
    SELECT 
        MIN(ts) as start_time,
        MAX(ts) as end_time
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
)

SELECT 
    start_time,
    end_time,
    CAST(start_time AS DATE) as start_date,
    CAST(end_time AS DATE) as end_date
FROM time_check;
