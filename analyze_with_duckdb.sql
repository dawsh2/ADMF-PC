-- Load and analyze the parquet file
SELECT 'Total rows:' as metric, COUNT(*) as value 
FROM read_parquet('/Users/daws/ADMF-PC/config/ensemble/results/20250623_084444/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
UNION ALL
SELECT 'Unique signal values:' as metric, COUNT(DISTINCT val) as value
FROM read_parquet('/Users/daws/ADMF-PC/config/ensemble/results/20250623_084444/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
UNION ALL
SELECT 'Min value:' as metric, MIN(val) as value
FROM read_parquet('/Users/daws/ADMF-PC/config/ensemble/results/20250623_084444/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
UNION ALL
SELECT 'Max value:' as metric, MAX(val) as value
FROM read_parquet('/Users/daws/ADMF-PC/config/ensemble/results/20250623_084444/traces/ensemble/SPY_5m_compiled_strategy_0.parquet');

-- Show distribution
SELECT val, COUNT(*) as count, 
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM read_parquet('/Users/daws/ADMF-PC/config/ensemble/results/20250623_084444/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
GROUP BY val
ORDER BY val;

-- Check for extreme values
SELECT 'Values outside [-1,0,1]:' as check_type, COUNT(*) as count
FROM read_parquet('/Users/daws/ADMF-PC/config/ensemble/results/20250623_084444/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
WHERE val < -1 OR val > 1;

-- Show sample of non-zero signals
SELECT time, val, strategy_index
FROM read_parquet('/Users/daws/ADMF-PC/config/ensemble/results/20250623_084444/traces/ensemble/SPY_5m_compiled_strategy_0.parquet')
WHERE val != 0
LIMIT 10;