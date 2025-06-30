-- Find strategy pairs with specific correlation characteristics
WITH signal_data AS (
    -- Get signals aligned by timestamp
    SELECT 
        strategy_hash,
        ts,
        val as signal_value,
        DATE_TRUNC('minute', ts) as minute_ts
    FROM signals
    WHERE val != 0
),
strategy_pairs AS (
    -- Create all possible pairs
    SELECT DISTINCT
        s1.strategy_hash as strategy1,
        s2.strategy_hash as strategy2
    FROM signal_data s1
    CROSS JOIN signal_data s2
    WHERE s1.strategy_hash < s2.strategy_hash
),
correlation_calc AS (
    -- Calculate correlation for each pair
    SELECT 
        sp.strategy1,
        sp.strategy2,
        COUNT(DISTINCT sd1.minute_ts) as common_periods,
        -- Correlation approximation using signal overlap
        SUM(CASE 
            WHEN sd1.signal_value * sd2.signal_value > 0 THEN 1 
            WHEN sd1.signal_value * sd2.signal_value < 0 THEN -1
            ELSE 0 
        END)::FLOAT / NULLIF(COUNT(*), 0) as signal_correlation
    FROM strategy_pairs sp
    LEFT JOIN signal_data sd1 
        ON sp.strategy1 = sd1.strategy_hash
    LEFT JOIN signal_data sd2 
        ON sp.strategy2 = sd2.strategy_hash 
        AND sd1.minute_ts = sd2.minute_ts
    GROUP BY sp.strategy1, sp.strategy2
    HAVING COUNT(*) >= {min_observations}
)
SELECT 
    c.strategy1,
    c.strategy2,
    s1.strategy_type as type1,
    s2.strategy_type as type2,
    c.signal_correlation,
    c.common_periods,
    s1.sharpe_ratio as sharpe1,
    s2.sharpe_ratio as sharpe2,
    -- Combined metrics
    (s1.sharpe_ratio + s2.sharpe_ratio) / 2 as avg_sharpe,
    s1.total_return + s2.total_return as combined_return
FROM correlation_calc c
JOIN strategies s1 ON c.strategy1 = s1.strategy_hash
JOIN strategies s2 ON c.strategy2 = s2.strategy_hash
WHERE ABS(c.signal_correlation) < {correlation_threshold}
    AND s1.sharpe_ratio > {min_sharpe}
    AND s2.sharpe_ratio > {min_sharpe}
ORDER BY avg_sharpe DESC