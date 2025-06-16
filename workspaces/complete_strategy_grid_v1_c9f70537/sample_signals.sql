
SELECT 
    ts,
    CAST(val AS INTEGER) as signal_val,
    px as price
FROM read_parquet('traces/SPY_1m/signals/macd_crossover_grid/SPY_macd_crossover_grid_5_20_11.parquet')
ORDER BY ts
LIMIT 30;
