-- 02_analyze_single_strategy.sql
-- Analyzes a single strategy's performance by regime
-- Requires: 01_setup_regime_analysis.sql to be run first
-- Parameters: strategy_path, strategy_id, strategy_name

PRAGMA memory_limit='2GB';

-- Parameters
SET VARIABLE strategy_path = getvariable('strategy_path');
SET VARIABLE strategy_id = COALESCE(getvariable('strategy_id'), 'unknown');
SET VARIABLE strategy_name = COALESCE(getvariable('strategy_name'), 'unknown');

-- Analyze strategy performance by regime
WITH 
strategy_signals AS (
    SELECT 
        ts::timestamp as signal_time,
        val as signal_value,
        LAG(val) OVER (ORDER BY ts::timestamp) as prev_signal,
        LEAD(ts::timestamp) OVER (ORDER BY ts::timestamp) as next_signal_time
    FROM read_parquet(getvariable('strategy_path'))
    WHERE ts::timestamp >= getvariable('start_date')::timestamp
      AND ts::timestamp <= getvariable('end_date')::timestamp
),
trades AS (
    SELECT 
        signal_time,
        signal_value,
        next_signal_time,
        EXTRACT(EPOCH FROM (next_signal_time - signal_time)) / 60.0 as duration_minutes
    FROM strategy_signals
    WHERE prev_signal IS NOT NULL
      AND prev_signal != signal_value
      AND next_signal_time IS NOT NULL
),
trades_with_data AS (
    SELECT 
        t.*,
        rt.current_regime as entry_regime,
        mp1.close as entry_price,
        mp2.close as exit_price,
        CASE 
            WHEN t.signal_value = 1 THEN (mp2.close - mp1.close) / mp1.close
            WHEN t.signal_value = -1 THEN (mp1.close - mp2.close) / mp1.close
        END as trade_return,
        CASE 
            WHEN t.signal_value = 1 THEN (mp2.close - mp1.close) / mp1.close - 0.0005
            WHEN t.signal_value = -1 THEN (mp1.close - mp2.close) / mp1.close - 0.0005
        END as net_return
    FROM trades t
    LEFT JOIN regime_timeline rt ON t.signal_time = rt.timestamp_est
    LEFT JOIN market_prices mp1 ON t.signal_time = mp1.timestamp_est
    LEFT JOIN market_prices mp2 ON t.next_signal_time = mp2.timestamp_est
    WHERE rt.current_regime IS NOT NULL
      AND mp1.close IS NOT NULL
      AND mp2.close IS NOT NULL
),
regime_performance AS (
    SELECT 
        entry_regime,
        COUNT(*) as trade_count,
        -- Gross returns
        AVG(trade_return) as avg_return,
        SUM(trade_return) as total_return,
        STDDEV(trade_return) as return_std,
        -- Net returns (after costs)
        AVG(net_return) as avg_net_return,
        SUM(net_return) as total_net_return,
        -- Risk metrics
        AVG(trade_return) / NULLIF(STDDEV(trade_return), 0) as sharpe_ratio,
        SUM(CASE WHEN trade_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
        -- Trade characteristics
        AVG(duration_minutes) as avg_duration_min,
        MIN(duration_minutes) as min_duration_min,
        MAX(duration_minutes) as max_duration_min,
        SUM(CASE WHEN signal_value = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as long_pct,
        -- Best/worst trades
        MAX(trade_return) as best_trade,
        MIN(trade_return) as worst_trade
    FROM trades_with_data
    GROUP BY entry_regime
)
SELECT 
    getvariable('strategy_id') as strategy_id,
    getvariable('strategy_name') as strategy_name,
    entry_regime,
    trade_count,
    ROUND(avg_return * 100, 3) as avg_return_pct,
    ROUND(total_return * 100, 3) as gross_return_pct,
    ROUND(total_net_return * 100, 3) as net_return_pct,
    ROUND(sharpe_ratio, 3) as sharpe_ratio,
    ROUND(win_rate, 1) as win_rate_pct,
    ROUND(long_pct, 1) as long_pct,
    ROUND(avg_duration_min, 1) as avg_duration_min,
    ROUND(min_duration_min, 1) as min_duration_min,
    ROUND(max_duration_min, 1) as max_duration_min,
    ROUND(best_trade * 100, 2) as best_trade_pct,
    ROUND(worst_trade * 100, 2) as worst_trade_pct
FROM regime_performance
WHERE trade_count >= 1
ORDER BY net_return_pct DESC;