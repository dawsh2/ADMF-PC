
SELECT ts, open, high, low, close, volume
FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
WHERE ts >= '2024-03-26' AND ts <= '2024-03-27'
ORDER BY ts
LIMIT 5;
