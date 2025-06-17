-- Cost-Adjusted Strategy Analysis with 0.5 bps transaction costs
-- Using individual parquet files to calculate net returns after execution costs

PRAGMA memory_limit='3GB';
SET threads=4;

-- Use the strategy scores as our base since it has regime-specific performance
-- This gives us a quick assessment of cost impact on the best strategies
WITH cost_impact AS (
    SELECT 
        strategy_name,
        strategy_type,
        current_regime,
        sharpe_ratio as gross_sharpe,
        total_return,
        win_rate,
        trading_days,
        
        -- Estimate trade frequency from total return and win rate
        -- Assuming average return per trade is total_return / estimated_trades
        CASE 
            WHEN win_rate > 0 AND total_return > 0 
            THEN ROUND(trading_days * 2.0)  -- Estimate 2 trades per trading day on average
            ELSE 100  -- Default minimum
        END as estimated_trades,
        
        -- Calculate estimated daily return
        total_return / NULLIF(trading_days, 0) as avg_daily_return_pct,
        
        -- Estimate cost impact: 0.5 bps per trade
        -- If we have 2 trades per day, that's 1 bps cost per day
        CASE 
            WHEN trading_days > 0 
            THEN (ROUND(trading_days * 2.0) * 0.5 / 10000) / trading_days  -- Daily cost in percentage
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
        
        -- Net Sharpe estimate (rough approximation)
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
SELECT 
    strategy_type,
    strategy_name,
    current_regime,
    estimated_trades,
    
    -- Performance metrics
    gross_sharpe,
    estimated_net_sharpe,
    ROUND(estimated_net_sharpe - gross_sharpe, 2) as sharpe_degradation,
    
    -- Return analysis  
    gross_daily_return_bps,
    daily_cost_bps,
    net_daily_return_bps,
    cost_impact_pct,
    
    status,
    win_rate_pct
FROM cost_adjusted_performance
ORDER BY estimated_net_sharpe DESC NULLS LAST;

-- Summary statistics
SELECT 
    '=== COST IMPACT SUMMARY ===' as summary_type;

SELECT 
    status,
    COUNT(*) as strategy_count,
    ROUND(AVG(cost_impact_pct), 1) as avg_cost_impact_pct,
    ROUND(AVG(gross_sharpe), 2) as avg_gross_sharpe,
    ROUND(AVG(estimated_net_sharpe), 2) as avg_net_sharpe
FROM cost_adjusted_performance
GROUP BY status
ORDER BY avg_net_sharpe DESC NULLS LAST;