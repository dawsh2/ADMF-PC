"""
Reusable DuckDB queries for strategy analysis.

Each function returns a query string that can be executed with parameters.
"""

def get_signal_frequency_query():
    """Query to analyze signal frequency by strategy"""
    return """
    WITH signal_stats AS (
        SELECT 
            strategy_hash,
            COUNT(*) as total_signals,
            COUNT(DISTINCT DATE(ts)) as trading_days,
            MIN(ts) as first_signal,
            MAX(ts) as last_signal,
            SUM(CASE WHEN val > 0 THEN 1 ELSE 0 END) as long_signals,
            SUM(CASE WHEN val < 0 THEN 1 ELSE 0 END) as short_signals
        FROM read_parquet('{trace_path}')
        WHERE val != 0
        GROUP BY strategy_hash
    )
    SELECT 
        s.*,
        s.total_signals::FLOAT / s.trading_days as signals_per_day,
        s.long_signals::FLOAT / NULLIF(s.total_signals, 0) as long_ratio,
        EXTRACT(EPOCH FROM (last_signal - first_signal)) / 86400.0 as active_days
    FROM signal_stats s
    ORDER BY signals_per_day DESC
    """


def get_intraday_pattern_query():
    """Query to find intraday trading patterns"""
    return """
    SELECT 
        EXTRACT(HOUR FROM ts) as hour,
        EXTRACT(DOW FROM ts) as day_of_week,
        COUNT(*) as signal_count,
        AVG(CASE WHEN val > 0 THEN 1 WHEN val < 0 THEN -1 ELSE 0 END) as avg_direction,
        COUNT(DISTINCT strategy_hash) as strategies_active
    FROM read_parquet('{trace_path}')
    WHERE val != 0
    GROUP BY hour, day_of_week
    ORDER BY hour, day_of_week
    """


def get_parameter_performance_query():
    """Query to aggregate performance by parameter values"""
    return """
    SELECT 
        {param_column} as param_value,
        COUNT(DISTINCT strategy_hash) as strategy_count,
        AVG(sharpe_ratio) as avg_sharpe,
        STDDEV(sharpe_ratio) as std_sharpe,
        MAX(sharpe_ratio) as max_sharpe,
        AVG(total_return) as avg_return,
        AVG(max_drawdown) as avg_drawdown
    FROM '{performance_table}'
    WHERE {param_column} IS NOT NULL
    GROUP BY {param_column}
    HAVING COUNT(*) >= {min_count}
    ORDER BY avg_sharpe DESC
    """


def get_correlation_pairs_query():
    """Query to find highly correlated strategy pairs"""
    return """
    WITH strategy_pairs AS (
        SELECT 
            s1.strategy_hash as hash1,
            s2.strategy_hash as hash2,
            s1.strategy_type as type1,
            s2.strategy_type as type2,
            CORR(s1.signal, s2.signal) as correlation
        FROM strategy_signals s1
        CROSS JOIN strategy_signals s2
        WHERE s1.strategy_hash < s2.strategy_hash
    )
    SELECT * FROM strategy_pairs
    WHERE ABS(correlation) > {correlation_threshold}
    ORDER BY ABS(correlation) DESC
    LIMIT {limit}
    """


def get_regime_performance_query():
    """Query to analyze performance by market regime"""
    return """
    WITH regime_stats AS (
        SELECT 
            r.regime_type,
            s.strategy_hash,
            s.strategy_type,
            COUNT(*) as signals_in_regime,
            SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) as winning_trades,
            AVG(trade_return) as avg_return_in_regime,
            STDDEV(trade_return) as return_volatility
        FROM signals s
        JOIN regimes r ON DATE(s.ts) = DATE(r.date)
        WHERE s.val != 0
        GROUP BY r.regime_type, s.strategy_hash, s.strategy_type
    )
    SELECT 
        regime_type,
        strategy_type,
        COUNT(DISTINCT strategy_hash) as strategies,
        AVG(avg_return_in_regime) as avg_return,
        AVG(winning_trades::FLOAT / NULLIF(signals_in_regime, 0)) as win_rate
    FROM regime_stats
    GROUP BY regime_type, strategy_type
    ORDER BY regime_type, avg_return DESC
    """


def get_burst_detection_query():
    """Query to detect signal bursts (rapid trading)"""
    return """
    WITH signal_gaps AS (
        SELECT 
            strategy_hash,
            ts,
            LAG(ts) OVER (PARTITION BY strategy_hash ORDER BY ts) as prev_ts,
            EXTRACT(EPOCH FROM (ts - LAG(ts) OVER (PARTITION BY strategy_hash ORDER BY ts))) / 3600 as hours_between
        FROM read_parquet('{trace_path}')
        WHERE val != 0
    ),
    burst_periods AS (
        SELECT 
            strategy_hash,
            DATE(ts) as burst_date,
            COUNT(*) as signals_in_day,
            MIN(hours_between) as min_gap_hours,
            AVG(hours_between) as avg_gap_hours
        FROM signal_gaps
        WHERE hours_between IS NOT NULL
        GROUP BY strategy_hash, DATE(ts)
        HAVING COUNT(*) >= {min_signals_per_burst}
    )
    SELECT 
        strategy_hash,
        COUNT(*) as burst_days,
        AVG(signals_in_day) as avg_signals_per_burst,
        MIN(min_gap_hours) as fastest_signal_gap
    FROM burst_periods
    GROUP BY strategy_hash
    ORDER BY burst_days DESC
    """


def get_drawdown_periods_query():
    """Query to analyze drawdown periods"""
    return """
    WITH drawdown_periods AS (
        SELECT 
            strategy_hash,
            start_date,
            end_date,
            drawdown_pct,
            recovery_days,
            CASE 
                WHEN drawdown_pct <= -0.05 THEN 'severe'
                WHEN drawdown_pct <= -0.03 THEN 'moderate'
                ELSE 'mild'
            END as severity
        FROM drawdowns
        WHERE drawdown_pct < {max_drawdown_threshold}
    )
    SELECT 
        strategy_hash,
        severity,
        COUNT(*) as occurrence_count,
        AVG(drawdown_pct) as avg_drawdown,
        AVG(recovery_days) as avg_recovery_days,
        MAX(drawdown_pct) as worst_drawdown
    FROM drawdown_periods
    GROUP BY strategy_hash, severity
    ORDER BY strategy_hash, severity
    """