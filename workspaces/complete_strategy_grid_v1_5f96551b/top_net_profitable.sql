-- Top 20 Net Profitable Strategies After 0.5 bps Transaction Costs
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
            THEN ROUND(trading_days * 2.0)  
            ELSE 100  
        END as estimated_trades,
        
        -- Calculate estimated daily return
        total_return / NULLIF(trading_days, 0) as avg_daily_return_pct,
        
        -- Estimate cost impact: 0.5 bps per trade
        CASE 
            WHEN trading_days > 0 
            THEN (ROUND(trading_days * 2.0) * 0.5 / 10000) / trading_days
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
        strategy_type,
        strategy_name,
        current_regime,
        estimated_trades,
        
        -- Original metrics
        ROUND(gross_sharpe, 2) as gross_sharpe,
        ROUND(win_rate * 100, 1) as win_rate_pct,
        
        -- Cost impact
        ROUND(avg_daily_return_pct * 10000, 2) as gross_daily_return_bps,
        ROUND(estimated_daily_cost_pct * 10000, 2) as daily_cost_bps,
        ROUND(net_daily_return_pct * 10000, 2) as net_daily_return_bps,
        
        -- Net Sharpe estimate 
        ROUND(net_daily_return_pct / NULLIF(avg_daily_return_pct, 0) * gross_sharpe, 2) as estimated_net_sharpe,
        
        -- Cost impact percentage
        ROUND(estimated_daily_cost_pct / NULLIF(ABS(avg_daily_return_pct), 0) * 100, 1) as cost_impact_pct
    FROM cost_impact
    WHERE net_daily_return_pct > 0  -- Only net profitable
)
SELECT 
    strategy_type,
    strategy_name,
    current_regime,
    estimated_net_sharpe,
    gross_daily_return_bps,
    daily_cost_bps,
    net_daily_return_bps,
    cost_impact_pct,
    win_rate_pct
FROM cost_adjusted_performance
ORDER BY estimated_net_sharpe DESC
LIMIT 20;