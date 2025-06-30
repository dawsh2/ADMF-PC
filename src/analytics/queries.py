"""
Pre-built DuckDB Queries for ADMF-PC Analytics

Collection of parameterized SQL queries for common trace analysis tasks.
Designed to work with sparse signal storage format (idx, val, px).
"""

class TraceQueries:
    """Pre-built DuckDB queries for trace analysis."""
    
    # Basic signal analysis
    TOP_PERFORMERS = """
    WITH signal_metrics AS (
        SELECT 
            strategy_id,
            COUNT(DISTINCT bar_idx) as signal_count,
            SUM(CASE WHEN signal_value > 0 THEN 1 ELSE 0 END) as long_signals,
            SUM(CASE WHEN signal_value < 0 THEN 1 ELSE 0 END) as short_signals,
            AVG(ABS(signal_value)) as avg_signal_strength,
            MAX(bar_idx) - MIN(bar_idx) as active_bars
        FROM signals
        GROUP BY strategy_id
    ),
    trades AS (
        SELECT 
            strategy_id,
            COUNT(*) / 2 as trade_count,  -- Entry + exit pairs
            AVG(return_pct) as avg_return,
            STDDEV(return_pct) as return_std,
            SUM(CASE WHEN return_pct > 0 THEN 1 ELSE 0 END) / (COUNT(*) / 2) as win_rate
        FROM (
            SELECT *,
                LAG(signal_value) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as prev_signal,
                LAG(price) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as prev_price,
                CASE 
                    WHEN signal_value = 0 AND LAG(signal_value) OVER (PARTITION BY strategy_id ORDER BY bar_idx) != 0 
                    THEN (price - LAG(price) OVER (PARTITION BY strategy_id ORDER BY bar_idx)) / LAG(price) OVER (PARTITION BY strategy_id ORDER BY bar_idx)
                    ELSE NULL
                END as return_pct
            FROM signals
        ) t
        WHERE return_pct IS NOT NULL
        GROUP BY strategy_id
    )
    SELECT 
        sm.strategy_id,
        sm.signal_count,
        sm.long_signals,
        sm.short_signals,
        COALESCE(t.trade_count, 0) as trade_count,
        COALESCE(t.avg_return, 0) as avg_return,
        COALESCE(t.win_rate, 0) as win_rate,
        CASE 
            WHEN COALESCE(t.return_std, 0) > 0 
            THEN t.avg_return / t.return_std * SQRT(252)  -- Annualized Sharpe
            ELSE 0 
        END as sharpe_ratio
    FROM signal_metrics sm
    LEFT JOIN trades t ON sm.strategy_id = t.strategy_id
    ORDER BY sharpe_ratio DESC
    LIMIT {limit}
    """
    
    # Filter effectiveness analysis
    FILTER_COMPARISON = """
    WITH strategy_filters AS (
        SELECT DISTINCT 
            strategy_id,
            -- Extract filter info from metadata (would need to join with metadata)
            CASE 
                WHEN strategy_id IN {baseline_ids} THEN 'baseline'
                ELSE 'filtered'
            END as filter_group
        FROM signals
    ),
    signal_counts AS (
        SELECT 
            sf.filter_group,
            s.strategy_id,
            COUNT(*) as signal_count
        FROM signals s
        JOIN strategy_filters sf ON s.strategy_id = sf.strategy_id
        GROUP BY sf.filter_group, s.strategy_id
    )
    SELECT 
        filter_group,
        COUNT(DISTINCT strategy_id) as num_strategies,
        AVG(signal_count) as avg_signals,
        MIN(signal_count) as min_signals,
        MAX(signal_count) as max_signals,
        STDDEV(signal_count) as signal_stddev
    FROM signal_counts
    GROUP BY filter_group
    """
    
    # Pattern frequency analysis
    SIGNAL_PATTERNS = """
    WITH signal_sequences AS (
        SELECT 
            strategy_id,
            bar_idx,
            signal_value,
            -- Create pattern string from window of signals
            STRING_AGG(
                CASE 
                    WHEN signal_value > 0 THEN 'L'
                    WHEN signal_value < 0 THEN 'S'
                    ELSE 'N'
                END,
                ''
            ) OVER (
                PARTITION BY strategy_id 
                ORDER BY bar_idx 
                ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW
            ) as pattern
        FROM signals
    ),
    pattern_counts AS (
        SELECT 
            pattern,
            COUNT(*) as occurrences,
            COUNT(DISTINCT strategy_id) as strategies_with_pattern,
            AVG(signal_value) as avg_signal_when_pattern
        FROM signal_sequences
        WHERE LENGTH(pattern) = {window_size} + 1  -- Full window only
        GROUP BY pattern
        HAVING COUNT(*) >= {min_occurrences}
    )
    SELECT 
        pattern,
        occurrences,
        strategies_with_pattern,
        avg_signal_when_pattern,
        occurrences::FLOAT / (SELECT COUNT(*) FROM signal_sequences WHERE LENGTH(pattern) = {window_size} + 1) as frequency
    FROM pattern_counts
    ORDER BY occurrences DESC
    LIMIT {limit}
    """
    
    # Trade analysis from sparse signals
    TRADE_EXTRACTION = """
    WITH signal_changes AS (
        SELECT 
            strategy_id,
            bar_idx,
            signal_value,
            price,
            LAG(signal_value, 1, 0) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as prev_signal,
            -- Detect signal transitions
            CASE 
                WHEN signal_value != 0 AND LAG(signal_value, 1, 0) OVER (PARTITION BY strategy_id ORDER BY bar_idx) = 0 THEN 'entry'
                WHEN signal_value = 0 AND LAG(signal_value, 1, 0) OVER (PARTITION BY strategy_id ORDER BY bar_idx) != 0 THEN 'exit'
                WHEN signal_value != 0 AND LAG(signal_value, 1, 0) OVER (PARTITION BY strategy_id ORDER BY bar_idx) != 0 
                     AND SIGN(signal_value) != SIGN(LAG(signal_value, 1, 0) OVER (PARTITION BY strategy_id ORDER BY bar_idx)) THEN 'reversal'
                ELSE 'hold'
            END as transition_type,
            ROW_NUMBER() OVER (PARTITION BY strategy_id ORDER BY bar_idx) as signal_seq
        FROM signals
        WHERE strategy_id IN ({strategy_ids})
    ),
    entries AS (
        SELECT 
            strategy_id,
            bar_idx as entry_bar,
            price as entry_price,
            signal_value as entry_signal,
            signal_seq as entry_seq
        FROM signal_changes
        WHERE transition_type IN ('entry', 'reversal')
    ),
    exits AS (
        SELECT 
            strategy_id,
            bar_idx as exit_bar,
            price as exit_price,
            signal_seq as exit_seq
        FROM signal_changes
        WHERE transition_type IN ('exit', 'reversal')
    )
    SELECT 
        e.strategy_id,
        e.entry_bar,
        e.entry_price,
        e.entry_signal,
        x.exit_bar,
        x.exit_price,
        x.exit_bar - e.entry_bar as duration_bars,
        CASE 
            WHEN e.entry_signal > 0 THEN 'long'
            ELSE 'short'
        END as direction,
        CASE 
            WHEN e.entry_signal > 0 THEN (x.exit_price - e.entry_price) / e.entry_price
            ELSE (e.entry_price - x.exit_price) / e.entry_price
        END as return_pct,
        CASE 
            WHEN e.entry_signal > 0 THEN x.exit_price - e.entry_price
            ELSE e.entry_price - x.exit_price
        END as pnl_points
    FROM entries e
    INNER JOIN exits x 
        ON e.strategy_id = x.strategy_id 
        AND x.exit_seq = (
            SELECT MIN(exit_seq) 
            FROM exits x2 
            WHERE x2.strategy_id = e.strategy_id 
            AND x2.exit_seq > e.entry_seq
        )
    ORDER BY e.strategy_id, e.entry_bar
    """
    
    # Price level analysis
    PRICE_LEVEL_PERFORMANCE = """
    WITH price_buckets AS (
        SELECT 
            strategy_id,
            bar_idx,
            signal_value,
            price,
            NTILE({n_buckets}) OVER (ORDER BY price) as price_bucket,
            MIN(price) OVER (PARTITION BY NTILE({n_buckets}) OVER (ORDER BY price)) as bucket_min,
            MAX(price) OVER (PARTITION BY NTILE({n_buckets}) OVER (ORDER BY price)) as bucket_max
        FROM signals
        WHERE signal_value != 0
    )
    SELECT 
        price_bucket,
        ROUND(AVG(bucket_min), 2) as price_range_start,
        ROUND(AVG(bucket_max), 2) as price_range_end,
        COUNT(*) as signal_count,
        SUM(CASE WHEN signal_value > 0 THEN 1 ELSE 0 END) as long_signals,
        SUM(CASE WHEN signal_value < 0 THEN 1 ELSE 0 END) as short_signals,
        AVG(ABS(signal_value)) as avg_signal_strength,
        COUNT(DISTINCT strategy_id) as strategies_active
    FROM price_buckets
    GROUP BY price_bucket
    ORDER BY price_bucket
    """
    
    # Time-based analysis
    TIME_OF_DAY_ANALYSIS = """
    WITH time_signals AS (
        SELECT 
            strategy_id,
            bar_idx,
            signal_value,
            price,
            -- Assuming bar_idx can be mapped to time of day
            (bar_idx % {bars_per_day}) as intraday_bar,
            CASE 
                WHEN (bar_idx % {bars_per_day}) < {bars_per_day} * 0.25 THEN 'morning'
                WHEN (bar_idx % {bars_per_day}) < {bars_per_day} * 0.5 THEN 'midday'
                WHEN (bar_idx % {bars_per_day}) < {bars_per_day} * 0.75 THEN 'afternoon'
                ELSE 'close'
            END as session
        FROM signals
        WHERE signal_value != 0
    )
    SELECT 
        session,
        COUNT(*) as signal_count,
        AVG(ABS(signal_value)) as avg_signal_strength,
        SUM(CASE WHEN signal_value > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as long_ratio,
        COUNT(DISTINCT strategy_id) as active_strategies
    FROM time_signals
    GROUP BY session
    ORDER BY 
        CASE session
            WHEN 'morning' THEN 1
            WHEN 'midday' THEN 2
            WHEN 'afternoon' THEN 3
            WHEN 'close' THEN 4
        END
    """
    
    # Signal clustering
    SIGNAL_CLUSTERS = """
    WITH signal_features AS (
        SELECT 
            s1.strategy_id,
            s1.bar_idx,
            s1.signal_value,
            s1.price,
            -- Calculate local volatility
            STDDEV(s2.price) OVER (
                PARTITION BY s1.strategy_id 
                ORDER BY s1.bar_idx 
                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
            ) as local_volatility,
            -- Calculate local trend
            (s1.price - AVG(s2.price) OVER (
                PARTITION BY s1.strategy_id 
                ORDER BY s1.bar_idx 
                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
            )) / AVG(s2.price) OVER (
                PARTITION BY s1.strategy_id 
                ORDER BY s1.bar_idx 
                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
            ) as local_trend
        FROM signals s1
        JOIN signals s2 ON s1.strategy_id = s2.strategy_id
        WHERE s1.signal_value != 0
    )
    SELECT 
        CASE 
            WHEN local_volatility < PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY local_volatility) OVER () THEN 'low_vol'
            WHEN local_volatility < PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY local_volatility) OVER () THEN 'med_vol'
            ELSE 'high_vol'
        END as volatility_regime,
        CASE 
            WHEN local_trend < -0.01 THEN 'downtrend'
            WHEN local_trend > 0.01 THEN 'uptrend'
            ELSE 'sideways'
        END as trend_regime,
        COUNT(*) as signal_count,
        AVG(signal_value) as avg_signal_value,
        COUNT(DISTINCT strategy_id) as strategies
    FROM signal_features
    GROUP BY 1, 2
    ORDER BY signal_count DESC
    """
    
    # Parameter sensitivity
    PARAMETER_ANALYSIS = """
    -- This query assumes we can extract parameters from strategy metadata
    WITH strategy_params AS (
        SELECT 
            strategy_id,
            -- These would come from joining with metadata
            {param1_expr} as param1,
            {param2_expr} as param2
        FROM signals
        GROUP BY strategy_id
    ),
    strategy_performance AS (
        SELECT 
            s.strategy_id,
            COUNT(*) as trade_count,
            AVG(
                CASE 
                    WHEN s.signal_value = 0 AND LAG(s.signal_value) OVER (PARTITION BY s.strategy_id ORDER BY s.bar_idx) != 0 
                    THEN (s.price - LAG(s.price) OVER (PARTITION BY s.strategy_id ORDER BY s.bar_idx)) / LAG(s.price) OVER (PARTITION BY s.strategy_id ORDER BY s.bar_idx)
                    ELSE NULL
                END
            ) as avg_return
        FROM signals s
        GROUP BY s.strategy_id
    )
    SELECT 
        sp.param1,
        sp.param2,
        AVG(perf.avg_return) as avg_return,
        AVG(perf.trade_count) as avg_trades,
        COUNT(*) as strategy_count,
        STDDEV(perf.avg_return) as return_stability
    FROM strategy_params sp
    JOIN strategy_performance perf ON sp.strategy_id = perf.strategy_id
    GROUP BY sp.param1, sp.param2
    HAVING COUNT(*) >= {min_strategies}
    ORDER BY avg_return DESC
    """
    
    # Drawdown analysis
    DRAWDOWN_ANALYSIS = """
    WITH cumulative_returns AS (
        SELECT 
            strategy_id,
            bar_idx,
            price,
            signal_value,
            SUM(
                CASE 
                    WHEN signal_value = 0 AND LAG(signal_value) OVER (PARTITION BY strategy_id ORDER BY bar_idx) != 0 
                    THEN (price - LAG(price) OVER (PARTITION BY strategy_id ORDER BY bar_idx)) / LAG(price) OVER (PARTITION BY strategy_id ORDER BY bar_idx)
                    ELSE 0
                END
            ) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as cum_return
        FROM signals
    ),
    running_max AS (
        SELECT 
            *,
            MAX(1 + cum_return) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as running_peak
        FROM cumulative_returns
    )
    SELECT 
        strategy_id,
        MIN((1 + cum_return) / running_peak - 1) as max_drawdown,
        AVG((1 + cum_return) / running_peak - 1) as avg_drawdown,
        COUNT(CASE WHEN (1 + cum_return) / running_peak - 1 < -0.05 THEN 1 END) as drawdown_periods_5pct,
        COUNT(CASE WHEN (1 + cum_return) / running_peak - 1 < -0.10 THEN 1 END) as drawdown_periods_10pct
    FROM running_max
    GROUP BY strategy_id
    ORDER BY max_drawdown DESC
    """