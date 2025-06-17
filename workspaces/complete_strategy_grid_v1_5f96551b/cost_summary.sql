-- Cost Impact Summary Analysis
PRAGMA memory_limit='3GB';
SET threads=4;

WITH cost_impact AS (
    SELECT 
        strategy_name,
        strategy_type,
        current_regime,
        sharpe_ratio as gross_sharpe,
        total_return,
        win_rate,
        trading_days,
        
        -- Estimate trade frequency 
        CASE 
            WHEN win_rate > 0 AND total_return > 0 
            THEN ROUND(trading_days * 2.0)  -- Estimate 2 trades per trading day
            ELSE 100  -- Default minimum
        END as estimated_trades,
        
        -- Calculate estimated daily return
        total_return / NULLIF(trading_days, 0) as avg_daily_return_pct,
        
        -- Estimate cost impact: 0.5 bps per trade
        CASE 
            WHEN trading_days > 0 
            THEN (ROUND(trading_days * 2.0) * 0.5 / 10000) / trading_days  -- Daily cost
            ELSE 0
        END as estimated_daily_cost_pct,
        
        -- Net daily return after costs
        (total_return / NULLIF(trading_days, 0)) - 
        CASE 
            WHEN trading_days > 0 
            THEN (ROUND(trading_days * 2.0) * 0.5 / 10000) / trading_days
            ELSE 0
        END as net_daily_return_pct
    FROM analytics.strategy_scores
    WHERE sharpe_ratio IS NOT NULL 
        AND total_return IS NOT NULL
        AND trading_days >= 20
),
cost_adjusted_performance AS (
    SELECT 
        strategy_name,
        strategy_type,
        current_regime,
        trading_days,
        estimated_trades,
        
        -- Original metrics
        ROUND(gross_sharpe, 2) as gross_sharpe,
        ROUND(total_return * 100, 2) as total_return_pct,
        ROUND(win_rate * 100, 1) as win_rate_pct,
        
        -- Cost impact
        ROUND(avg_daily_return_pct * 10000, 2) as gross_daily_return_bps,
        ROUND(estimated_daily_cost_pct * 10000, 2) as daily_cost_bps,
        ROUND(net_daily_return_pct * 10000, 2) as net_daily_return_bps,
        
        -- Net Sharpe estimate 
        ROUND(net_daily_return_pct / NULLIF(avg_daily_return_pct, 0) * gross_sharpe, 2) as estimated_net_sharpe,
        
        -- Profitability status
        CASE 
            WHEN net_daily_return_pct > 0 THEN 'NET_PROFITABLE'
            WHEN avg_daily_return_pct > 0 THEN 'GROSS_PROFITABLE_NET_UNPROFITABLE'
            ELSE 'UNPROFITABLE'
        END as status,
        
        -- Cost impact percentage
        ROUND(estimated_daily_cost_pct / NULLIF(ABS(avg_daily_return_pct), 0) * 100, 1) as cost_impact_pct
    FROM cost_impact
)
-- Summary statistics
SELECT 
    status,
    COUNT(*) as strategy_count,
    ROUND(AVG(cost_impact_pct), 1) as avg_cost_impact_pct,
    ROUND(AVG(gross_sharpe), 2) as avg_gross_sharpe,
    ROUND(AVG(estimated_net_sharpe), 2) as avg_net_sharpe,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM cost_adjusted_performance), 1) as percentage_of_total
FROM cost_adjusted_performance
GROUP BY status
ORDER BY avg_net_sharpe DESC NULLS LAST;