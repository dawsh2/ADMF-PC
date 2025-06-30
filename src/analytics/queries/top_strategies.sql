-- Top performing strategies with comprehensive metrics
SELECT 
    s.strategy_hash,
    s.strategy_type,
    s.sharpe_ratio,
    s.total_return,
    s.max_drawdown,
    s.win_rate,
    s.total_trades,
    s.avg_trade_duration,
    -- Calculate return per unit of risk
    s.total_return / ABS(s.max_drawdown) as return_drawdown_ratio,
    -- Profitability metrics
    s.total_return / NULLIF(s.total_trades, 0) as avg_return_per_trade,
    -- Extract parameters for analysis
    s.param_names,
    s.param_values
FROM strategies s
WHERE s.sharpe_ratio > {min_sharpe}
    AND s.total_trades >= {min_trades}
    AND s.max_drawdown > {max_drawdown_limit}
ORDER BY s.sharpe_ratio DESC
LIMIT {limit}