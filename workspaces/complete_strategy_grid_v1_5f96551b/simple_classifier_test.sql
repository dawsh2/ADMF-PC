-- Simple test to check classifier data structure
PRAGMA memory_limit='3GB';

-- Test reading one classifier file
SELECT 
    COUNT(*) as total_records,
    MIN(ts) as min_time,
    MAX(ts) as max_time,
    COUNT(DISTINCT val) as unique_states
FROM read_parquet('/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/traces/SPY_1m/classifiers/hidden_markov_grid/SPY_hidden_markov_grid_11_0001_05.parquet');