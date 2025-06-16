-- TEMPLATE: Correct Sharpe Ratio Calculation Using Wall-Clock Time
-- Prevents artificial inflation of Sharpe ratios for low-frequency strategies

PRAGMA memory_limit='3GB';
SET threads=4;

-- Replace with your actual database path
-- ATTACH 'path/to/analytics.duckdb' AS analytics;

WITH trades AS (
    -- REPLACE THIS SECTION with your trade construction logic
    SELECT 
        strategy_id,
        entry_time,
        net_return_bps  -- Should include transaction costs (e.g., gross_return - 0.5)
    FROM your_trades_table
    WHERE net_return_bps IS NOT NULL
),
daily_returns AS (
    -- Aggregate trades by day
    SELECT 
        strategy_id,
        DATE(entry_time) as trade_date,
        SUM(net_return_bps) as daily_net_bps,
        COUNT(*) as trades_that_day
    FROM trades
    GROUP BY strategy_id, DATE(entry_time)
),
date_ranges AS (
    -- Get trading period for each strategy
    SELECT 
        strategy_id,
        MIN(trade_date) as first_trade_date,
        MAX(trade_date) as last_trade_date
    FROM daily_returns
    GROUP BY strategy_id
),
all_trading_days AS (
    -- Generate complete date series (including zero-return days)
    SELECT 
        d.strategy_id,
        generate_series(d.first_trade_date, d.last_trade_date, INTERVAL 1 DAY)::DATE as calendar_date
    FROM date_ranges d
),
complete_daily_series AS (
    -- Include zero returns for days with no trades
    SELECT 
        a.strategy_id,
        a.calendar_date,
        COALESCE(r.daily_net_bps, 0) as daily_net_bps,
        COALESCE(r.trades_that_day, 0) as trades_that_day
    FROM all_trading_days a
    LEFT JOIN daily_returns r ON a.strategy_id = r.strategy_id 
        AND a.calendar_date = r.trade_date
    WHERE EXTRACT(DOW FROM a.calendar_date) BETWEEN 1 AND 5  -- Weekdays only
)
-- Calculate corrected metrics
SELECT 
    strategy_id,
    COUNT(*) as total_trading_days,
    SUM(trades_that_day) as total_trades,
    ROUND(SUM(trades_that_day)::DOUBLE / COUNT(*), 2) as trades_per_day,
    ROUND(AVG(daily_net_bps), 3) as avg_daily_return_bps,
    ROUND(STDDEV(daily_net_bps), 3) as daily_volatility_bps,
    -- CORRECT Sharpe: based on daily returns over wall-clock time
    ROUND(AVG(daily_net_bps) / NULLIF(STDDEV(daily_net_bps), 0) * SQRT(252), 2) as correct_annual_sharpe,
    CASE 
        WHEN AVG(daily_net_bps) > 0 THEN 'PROFITABLE'
        ELSE 'UNPROFITABLE'
    END as status
FROM complete_daily_series
GROUP BY strategy_id
HAVING SUM(trades_that_day) >= 100  -- Minimum trades for statistical significance
   AND COUNT(*) >= 20               -- Minimum trading days
ORDER BY correct_annual_sharpe DESC;