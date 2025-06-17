-- Cost-Adjusted Strategy Analysis with 0.5 bps transaction costs
-- Realistic assessment of strategy profitability after execution costs

PRAGMA memory_limit='3GB';
SET threads=4;

ATTACH '/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_5f96551b/analytics.duckdb' AS analytics;

WITH trades AS (
    SELECT 
        s.strategy_type || '_' || s.strategy_name as strategy_id,
        s.entry_time,
        s.exit_time,
        s.symbol_timeframe,
        s.signal_value,
        -- Calculate gross return in basis points
        CASE 
            WHEN s.signal_value = 1 THEN (s.exit_price - s.entry_price) / s.entry_price * 10000
            WHEN s.signal_value = -1 THEN (s.entry_price - s.exit_price) / s.entry_price * 10000
            ELSE 0
        END as gross_return_bps,
        -- Apply 0.5 bps transaction cost per trade
        CASE 
            WHEN s.signal_value = 1 THEN (s.exit_price - s.entry_price) / s.entry_price * 10000 - 0.5
            WHEN s.signal_value = -1 THEN (s.entry_price - s.exit_price) / s.entry_price * 10000 - 0.5
            ELSE 0
        END as net_return_bps
    FROM analytics.strategy_signals s
    WHERE s.entry_price IS NOT NULL 
        AND s.exit_price IS NOT NULL
        AND s.signal_value IN (1, -1)
        AND s.entry_time >= '2024-03-26'
        AND s.entry_time <= '2025-01-17'
),
daily_returns AS (
    SELECT 
        strategy_id,
        DATE(entry_time) as trade_date,
        SUM(gross_return_bps) as daily_gross_bps,
        SUM(net_return_bps) as daily_net_bps,
        COUNT(*) as trades_that_day
    FROM trades
    GROUP BY strategy_id, DATE(entry_time)
),
date_ranges AS (
    SELECT 
        strategy_id,
        MIN(trade_date) as first_trade_date,
        MAX(trade_date) as last_trade_date
    FROM daily_returns
    GROUP BY strategy_id
),
all_trading_days AS (
    SELECT 
        d.strategy_id,
        generate_series(d.first_trade_date, d.last_trade_date, INTERVAL 1 DAY)::DATE as calendar_date
    FROM date_ranges d
),
complete_daily_series AS (
    SELECT 
        a.strategy_id,
        a.calendar_date,
        COALESCE(r.daily_gross_bps, 0) as daily_gross_bps,
        COALESCE(r.daily_net_bps, 0) as daily_net_bps,
        COALESCE(r.trades_that_day, 0) as trades_that_day
    FROM all_trading_days a
    LEFT JOIN daily_returns r ON a.strategy_id = r.strategy_id 
        AND a.calendar_date = r.trade_date
    WHERE EXTRACT(DOW FROM a.calendar_date) BETWEEN 1 AND 5  -- Weekdays only
),
strategy_performance AS (
    SELECT 
        strategy_id,
        COUNT(*) as total_trading_days,
        SUM(trades_that_day) as total_trades,
        ROUND(SUM(trades_that_day)::DOUBLE / COUNT(*), 2) as trades_per_day,
        
        -- GROSS performance metrics
        ROUND(AVG(daily_gross_bps), 3) as avg_daily_gross_bps,
        ROUND(STDDEV(daily_gross_bps), 3) as daily_gross_volatility_bps,
        ROUND(AVG(daily_gross_bps) / NULLIF(STDDEV(daily_gross_bps), 0) * SQRT(252), 2) as gross_annual_sharpe,
        
        -- NET performance metrics (after 0.5 bps costs)
        ROUND(AVG(daily_net_bps), 3) as avg_daily_net_bps,
        ROUND(STDDEV(daily_net_bps), 3) as daily_net_volatility_bps,
        ROUND(AVG(daily_net_bps) / NULLIF(STDDEV(daily_net_bps), 0) * SQRT(252), 2) as net_annual_sharpe,
        
        -- Cost impact analysis
        ROUND(AVG(daily_gross_bps) - AVG(daily_net_bps), 3) as avg_daily_cost_bps,
        ROUND((AVG(daily_gross_bps) - AVG(daily_net_bps)) / NULLIF(AVG(daily_gross_bps), 0) * 100, 1) as cost_impact_percent,
        
        CASE 
            WHEN AVG(daily_net_bps) > 0 THEN 'NET_PROFITABLE'
            WHEN AVG(daily_gross_bps) > 0 THEN 'GROSS_PROFITABLE_NET_UNPROFITABLE'
            ELSE 'UNPROFITABLE'
        END as profitability_status
    FROM complete_daily_series
    GROUP BY strategy_id
    HAVING SUM(trades_that_day) >= 100  -- Minimum trades for statistical significance
       AND COUNT(*) >= 20               -- Minimum trading days
)
SELECT 
    strategy_id,
    total_trades,
    trades_per_day,
    
    -- Performance comparison
    gross_annual_sharpe,
    net_annual_sharpe,
    ROUND(net_annual_sharpe - gross_annual_sharpe, 2) as sharpe_degradation,
    
    -- Return analysis
    avg_daily_gross_bps,
    avg_daily_net_bps,
    avg_daily_cost_bps,
    cost_impact_percent,
    
    profitability_status
FROM strategy_performance
ORDER BY net_annual_sharpe DESC;