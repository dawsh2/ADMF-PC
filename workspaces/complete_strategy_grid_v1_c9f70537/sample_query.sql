
-- First understand the data structure 
WITH classifier_sample AS (
    SELECT ts, val, strat
    FROM read_parquet('traces/SPY_1m/classifiers/volatility_momentum_grid/SPY_volatility_momentum_grid_12_70_20.parquet')
    LIMIT 10
),

strategy_sample AS (
    SELECT ts, val, px, strat  
    FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
    LIMIT 10
)

SELECT 'CLASSIFIER_SAMPLE' as type, ts, val, strat, NULL as px FROM classifier_sample
UNION ALL
SELECT 'STRATEGY_SAMPLE' as type, ts, val, strat, px FROM strategy_sample
ORDER BY type, ts;
