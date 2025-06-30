-- Detect and analyze signal patterns across strategies
WITH signal_sequences AS (
    -- Get signal sequences with timing information
    SELECT 
        strategy_hash,
        ts,
        val as signal,
        -- Previous and next signals
        LAG(val, 1) OVER (PARTITION BY strategy_hash ORDER BY ts) as prev_signal,
        LEAD(val, 1) OVER (PARTITION BY strategy_hash ORDER BY ts) as next_signal,
        -- Time gaps
        EXTRACT(EPOCH FROM (ts - LAG(ts) OVER (PARTITION BY strategy_hash ORDER BY ts))) / 3600 as hours_since_last,
        EXTRACT(EPOCH FROM (LEAD(ts) OVER (PARTITION BY strategy_hash ORDER BY ts) - ts)) / 3600 as hours_to_next,
        -- Signal changes
        CASE 
            WHEN val != LAG(val) OVER (PARTITION BY strategy_hash ORDER BY ts) THEN 1 
            ELSE 0 
        END as signal_change,
        -- Time of day info
        EXTRACT(HOUR FROM ts) as hour_of_day,
        EXTRACT(DOW FROM ts) as day_of_week
    FROM signals
    WHERE val != 0
),
pattern_stats AS (
    -- Calculate pattern statistics
    SELECT 
        strategy_hash,
        -- Signal frequency patterns
        COUNT(*) as total_signals,
        AVG(hours_since_last) as avg_hours_between_signals,
        STDDEV(hours_since_last) as std_hours_between_signals,
        MIN(hours_since_last) as min_gap,
        MAX(hours_since_last) as max_gap,
        
        -- Direction patterns
        SUM(CASE WHEN signal > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as long_ratio,
        SUM(signal_change)::FLOAT / COUNT(*) as reversal_frequency,
        
        -- Clustering patterns (signals within 1 hour of each other)
        SUM(CASE WHEN hours_since_last <= 1 THEN 1 ELSE 0 END) as clustered_signals,
        
        -- Time patterns
        MODE() WITHIN GROUP (ORDER BY hour_of_day) as most_active_hour,
        MODE() WITHIN GROUP (ORDER BY day_of_week) as most_active_day
    FROM signal_sequences
    WHERE hours_since_last IS NOT NULL
    GROUP BY strategy_hash
),
signal_bursts AS (
    -- Identify burst periods (3+ signals within 4 hours)
    SELECT 
        strategy_hash,
        ts as burst_start,
        COUNT(*) OVER (
            PARTITION BY strategy_hash 
            ORDER BY ts 
            RANGE BETWEEN CURRENT ROW AND '4 hours' FOLLOWING
        ) as signals_in_burst
    FROM signals
    WHERE val != 0
),
burst_summary AS (
    SELECT 
        strategy_hash,
        COUNT(CASE WHEN signals_in_burst >= 3 THEN 1 END) as burst_count,
        MAX(signals_in_burst) as max_burst_size,
        AVG(CASE WHEN signals_in_burst >= 3 THEN signals_in_burst END) as avg_burst_size
    FROM signal_bursts
    GROUP BY strategy_hash
)
SELECT 
    ps.*,
    st.strategy_type,
    st.sharpe_ratio,
    st.total_return,
    bs.burst_count,
    bs.max_burst_size,
    bs.avg_burst_size,
    -- Pattern scores
    ps.clustered_signals::FLOAT / ps.total_signals as clustering_score,
    CASE 
        WHEN ps.std_hours_between_signals < ps.avg_hours_between_signals * 0.5 THEN 'regular'
        WHEN ps.std_hours_between_signals > ps.avg_hours_between_signals * 1.5 THEN 'irregular'
        ELSE 'mixed'
    END as timing_pattern,
    CASE
        WHEN ps.reversal_frequency > 0.7 THEN 'high_reversal'
        WHEN ps.reversal_frequency < 0.3 THEN 'trending'
        ELSE 'balanced'
    END as signal_behavior
FROM pattern_stats ps
JOIN strategies st ON ps.strategy_hash = st.strategy_hash
LEFT JOIN burst_summary bs ON ps.strategy_hash = bs.strategy_hash
WHERE ps.total_signals >= {min_signals}
ORDER BY st.sharpe_ratio DESC