
-- Examine SPY source data structure
SELECT 
    'STRUCTURE' as analysis,
    column_name,
    column_type,
    NULL as sample_value
FROM (
    DESCRIBE SELECT * FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet') LIMIT 1
)

UNION ALL

-- Show sample data
SELECT 
    'SAMPLE' as analysis,
    CAST(ts AS VARCHAR) as column_name,
    CAST(close AS VARCHAR) as column_type,
    CAST(volume AS VARCHAR) as sample_value
FROM read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
ORDER BY ts
LIMIT 5

ORDER BY analysis DESC, column_name;
