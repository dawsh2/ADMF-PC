-- 04_calculate_daily_sharpe.sql
-- Calculates proper daily Sharpe ratios for strategies
-- Uses daily returns instead of per-trade returns
-- Requires: regime_timeline and market_prices tables

PRAGMA memory_limit='2GB';

-- Parameters
SET VARIABLE strategy_path = getvariable('strategy_path');
SET VARIABLE strategy_id = COALESCE(getvariable('strategy_id'), 'unknown');
SET VARIABLE strategy_name = COALESCE(getvariable('strategy_name'), 'unknown');
SET VARIABLE risk_free_rate = COALESCE(getvariable('risk_free_rate'), 0.0);

-- Calculate daily returns for the strategy
WITH 
-- Get all signals
strategy_signals AS (
    SELECT 
        ts::timestamp as signal_time,
        val as signal_value,
        LAG(val) OVER (ORDER BY ts::timestamp) as prev_signal
    FROM read_parquet(getvariable('strategy_path'))
    WHERE ts::timestamp >= getvariable('start_date')::timestamp
      AND ts::timestamp <= getvariable('end_date')::timestamp
),
-- Identify position changes
position_changes AS (
    SELECT 
        signal_time,
        signal_value as position,
        ROW_NUMBER() OVER (ORDER BY signal_time) as change_num
    FROM strategy_signals
    WHERE (prev_signal IS NULL OR prev_signal != signal_value)
),
-- Create position timeline (forward-fill positions)
position_timeline AS (
    SELECT 
        mp.timestamp_est,
        mp.close,
        rt.current_regime,
        LAST_VALUE(pc.position IGNORE NULLS) OVER (
            ORDER BY mp.timestamp_est 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as position
    FROM market_prices mp
    LEFT JOIN regime_timeline rt ON mp.timestamp_est = rt.timestamp_est
    LEFT JOIN position_changes pc ON mp.timestamp_est = pc.signal_time
),
-- Calculate minute-level returns
minute_returns AS (
    SELECT 
        timestamp_est,
        current_regime,
        position,
        close,
        LAG(close) OVER (ORDER BY timestamp_est) as prev_close,
        CASE 
            WHEN position = 1 THEN (close - LAG(close) OVER (ORDER BY timestamp_est)) / LAG(close) OVER (ORDER BY timestamp_est)
            WHEN position = -1 THEN (LAG(close) OVER (ORDER BY timestamp_est) - close) / LAG(close) OVER (ORDER BY timestamp_est)
            ELSE 0
        END as minute_return
    FROM position_timeline
    WHERE position IS NOT NULL
),
-- Aggregate to daily returns
daily_returns AS (
    SELECT 
        DATE_TRUNC('day', timestamp_est) as trading_day,
        current_regime,
        SUM(minute_return) as daily_return,
        -- Track position changes and trades
        COUNT(DISTINCT position) as position_changes,
        SUM(CASE WHEN position != 0 THEN 1 ELSE 0 END) as minutes_in_position
    FROM minute_returns
    WHERE prev_close IS NOT NULL
    GROUP BY DATE_TRUNC('day', timestamp_est), current_regime
),
-- Calculate Sharpe ratio by regime
regime_sharpe AS (
    SELECT 
        current_regime,
        COUNT(*) as trading_days,
        AVG(daily_return) as avg_daily_return,
        STDDEV(daily_return) as daily_volatility,
        -- Daily Sharpe ratio
        (AVG(daily_return) - getvariable('risk_free_rate')::double) / NULLIF(STDDEV(daily_return), 0) as daily_sharpe,
        -- Annualized Sharpe (assuming 252 trading days)
        (AVG(daily_return) - getvariable('risk_free_rate')::double) / NULLIF(STDDEV(daily_return), 0) * SQRT(252) as annualized_sharpe,
        -- Other metrics
        SUM(daily_return) as total_return,
        SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_days_pct,
        AVG(position_changes) as avg_trades_per_day
    FROM daily_returns
    GROUP BY current_regime
)
SELECT 
    getvariable('strategy_id') as strategy_id,
    getvariable('strategy_name') as strategy_name,
    current_regime,
    trading_days,
    ROUND(avg_daily_return * 100, 4) as avg_daily_return_pct,
    ROUND(daily_volatility * 100, 4) as daily_volatility_pct,
    ROUND(daily_sharpe, 3) as daily_sharpe_ratio,
    ROUND(annualized_sharpe, 3) as annualized_sharpe_ratio,
    ROUND(total_return * 100, 2) as total_return_pct,
    ROUND(win_days_pct, 1) as win_days_pct,
    ROUND(avg_trades_per_day, 1) as avg_trades_per_day
FROM regime_sharpe
WHERE trading_days >= 5  -- Minimum days for meaningful statistics
ORDER BY annualized_sharpe_ratio DESC;